//! This module contains types that can be used to implement atomic borrowing.
//!
//! [`AtomicBorrow`] is used to borrow immutably.\
//! [`AtomicBorrow::try_new`] locks an atomic in such a way
//! that it can be locked immutably again but not mutably.
//! It fails if already borrowed mutably.
//!
//! [`AtomicBorrowMut`] is used to borrow mutably.\
//! [`AtomicBorrowMut::try_new`] locks an atomic in such a way
//! that it cannot be locked again.
//! It fails if already borrowed.
//!
//! Both types use [`AtomicIsize`] as a locking atomic.\
//! Where `0` means "not borrowed",\
//! immutable borrows are represented by positive values\
//! and mutable borrows are represented by negative values.
//!
//! [`AtomicBorrow::try_new`]: struct.AtomicBorrow.html#method.try_new
//! [`AtomicBorrowMut::try_new`]: struct.AtomicBorrowMut.html#method.try_new
//!

use core::sync::atomic::{AtomicIsize, Ordering};

/// Limit on the count of concurrent [`Ref`] instances for the same [`AtomicCell`].
const REF_LIMIT_FLAG: isize = 1 + (isize::MAX >> 1);

/// Dummy lock value that is used to create [`Ref`] and [`RefMut`] from references instead of [`AtomicCell`].
static DUMMY_LOCK: AtomicIsize = AtomicIsize::new(0);

fn is_dummy(lock: &AtomicIsize) -> bool {
    core::ptr::eq(lock, &DUMMY_LOCK)
}

/// Returns true if lock with specified lock value is locked for reads.
#[inline(always)]
pub fn is_reading(v: isize) -> bool {
    v > 0
}

/// Returns true if lock with specified lock value is locked for writes.
#[inline(always)]
pub fn is_writing(v: isize) -> bool {
    v < 0
}

/// Returns true if there are too many read refs.
///
/// If `is_reading` would return `false` for provided argument,
/// this function result unspecified.
#[inline(always)]
pub fn check_read_refs_count(v: isize) -> bool {
    v & REF_LIMIT_FLAG == REF_LIMIT_FLAG
}

/// Returns true if there are too many write refs.
///
/// If `is_writing` would return `false` for provided argument,
/// this function result unspecified.
#[inline(always)]
pub fn check_write_refs_count(v: isize) -> bool {
    v & REF_LIMIT_FLAG == 0
}

/// Encapsulates shared borrowing state.
///
/// An instance of this type guarantees that [`AtomicBorrowMut`] cannot be constructed for the same lock.
#[repr(transparent)]
pub struct AtomicBorrow<'a> {
    lock: &'a AtomicIsize,
}

impl<'a> AtomicBorrow<'a> {
    /// Attempts to borrow lock immutably.
    ///
    /// Fails if `AtomicIsize` contains a value for which `is_writing` returns true.
    ///
    /// On success `AtomicIsize` contains value for which `is_reading` returns true.
    #[inline]
    pub fn try_new(lock: &'a AtomicIsize) -> Option<Self> {
        debug_assert!(!is_dummy(lock), "dummy lock cannot be used from outside");

        loop {
            // Get original value.
            let val = lock.load(Ordering::Relaxed);

            if is_writing(val) {
                // Locked for writing.
                return None;
            }

            // Ensure that counter won't overflow into writing state.
            // `REF_LIMIT_FLAG` allows plenty clones to be created.
            // In fact, without `forget`ing the `Ref` instances, the counter cannot overflow
            // because this much `Ref` instances won't fit into address space.
            // `REF_LIMIT_FLAG` value allows `REF_LIMIT_FLAG - 1` concurrent attempts to borrow immutably.
            // Which assumed to never happen as there can't be that much threads.
            if check_read_refs_count(val) {
                too_many_refs();
            }

            let ok = lock
                .compare_exchange_weak(val, val + 1, Ordering::Acquire, Ordering::Relaxed)
                .is_ok();

            if ok {
                // It is now safe to construct immutable borrow.
                return Some(AtomicBorrow { lock });
            }
        }
    }

    /// Returns dummy atomic borrow that doesn't actually locks anything.
    /// It is used within [`Ref::new`] method that take external reference.
    ///
    /// [`Ref::new`]: struct.Ref.html#method.new
    #[inline(always)]
    pub fn dummy() -> Self {
        AtomicBorrow { lock: &DUMMY_LOCK }
    }

    /// Borrows can be cloned.
    ///
    /// There is a hard limit on number of clones for each lock (except dummy lock) and it will panic if cloning fails.
    /// Huge limit value makes it practically unreachable without `forget`ing clones in a loop.
    #[inline]
    pub fn clone(&self) -> AtomicBorrow<'a> {
        if is_dummy(self.lock) {
            return AtomicBorrow { lock: &DUMMY_LOCK };
        }

        // Lock is already held by `Ref` instance,
        // so `Relaxed` ordering is fine.
        let old = self.lock.fetch_add(1, Ordering::Relaxed);

        // Check that counter won't overflow into writing state.
        // REF_LIMIT_FLAG allows plenty clones to be created.
        // In fact, without `forget`ing the `AtomicBorrow` instances, the counter cannot overflow
        // because this much `AtomicBorrow` instances won't fit into address space.
        // And there can be `REF_LIMIT_FLAG - 1` attempts to create excessive `AtomicBorrow`s
        // that assumed to never happen as there can't be that much threads.
        if check_read_refs_count(old) {
            // Forbid creating this much `AtomicBorrow` instances.
            // Balance lock increment with decrement.
            self.lock.fetch_sub(1, Ordering::Relaxed);

            // Panic is put into separate non-inlineable cold function
            // in order to not pollute this function
            // and hit compiler that this branch is unlikely.
            too_many_refs();
        } else {
            AtomicBorrow { lock: self.lock }
        }
    }
}

impl<'a> Drop for AtomicBorrow<'a> {
    #[inline(always)]
    fn drop(&mut self) {
        if is_dummy(self.lock) {
            return;
        }

        debug_assert!(is_dummy(self.lock) || is_reading(self.lock.load(Ordering::Relaxed)));

        // Reset the lock counter. `Release` semantics is required
        // to sync with `Acquire` semantics within `try_borrow` and `try_borrow_mut`.
        self.lock.fetch_sub(1, Ordering::Release);
    }
}

/// Encapsulates exclusive borrowing state.
///
/// An instance of this type guarantees that neither [`AtomicBorrowMut`], nor [`AtomicBorrow`] cannot be constructed for the same lock.
#[repr(transparent)]
pub struct AtomicBorrowMut<'a> {
    lock: &'a AtomicIsize,
}

impl<'a> AtomicBorrowMut<'a> {
    /// Attempts to borrow lock mutably.
    #[inline]
    pub fn try_new(lock: &'a AtomicIsize) -> Option<Self> {
        debug_assert!(!is_dummy(lock), "dummy lock cannot be used from outside");

        // `Acquire` semantics syncs this operation with `Release` semantics in `Ref::drop` and `RefMut::drop`.
        // Lock counter `0` ensures that no other borrows exist.
        let ok = lock
            .compare_exchange(0, -1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok();

        if ok {
            // It is now safe to construct mutable borrow.
            Some(AtomicBorrowMut { lock })
        } else {
            None
        }
    }

    /// Returns dummy atomic borrow that doesn't actually locks anything.
    /// It is used within [`RefMut::new`] method that take external reference.
    ///
    /// [`RefMut::new`]: struct.RefMut.html#method.new
    #[inline(always)]
    pub fn dummy() -> Self {
        AtomicBorrowMut { lock: &DUMMY_LOCK }
    }

    /// Borrows can be cloned.
    ///
    /// Mutable borrows are cloned when borrowed reference is split into disjoint parts.
    ///
    /// There is a hard limit on number of clones for each lock (except dummy lock) and it will panic if cloning fails.
    /// Huge limit value makes it practically unreachable without `forget`ing clones in a loop.
    #[inline]
    pub fn clone(&self) -> AtomicBorrowMut<'a> {
        if is_dummy(self.lock) {
            return AtomicBorrowMut { lock: &DUMMY_LOCK };
        }

        // Lock is already held by `RefMut` instance,
        // so `Relaxed` ordering is fine.
        let old = self.lock.fetch_sub(1, Ordering::Relaxed);

        // Check that counter won't overflow into reading state.
        // REF_LIMIT_FLAG allows plenty clones to be created.
        // In fact, without `forget`ing the `AtomicBorrowMut` instances, the counter cannot overflow
        // because this much `AtomicBorrowMut` instances won't fit into address space.
        // And there can be `REF_LIMIT_FLAG - 1` attempts to create excessive `AtomicBorrowMut`s
        // that assumed to never happen as there can't be that much threads.
        if check_write_refs_count(old) {
            // Forbid creating this much `AtomicBorrowMut` instances.
            // Balance lock decrement with increment.
            self.lock.fetch_add(1, Ordering::Relaxed);

            // Panic is put into separate non-inlineable cold function
            // in order to not pollute this function
            // and hit compiler that this branch is unlikely.
            too_many_refs();
        } else {
            AtomicBorrowMut { lock: self.lock }
        }
    }
}

impl<'a> Drop for AtomicBorrowMut<'a> {
    #[inline(always)]
    fn drop(&mut self) {
        if is_dummy(self.lock) {
            return;
        }

        debug_assert!(is_dummy(self.lock) || is_writing(self.lock.load(Ordering::Relaxed)));

        // Increment lock counter. `Release` semantics is required
        // to sync with `Acquire` semantics within `try_borrow` and `try_borrow_mut`.
        self.lock.fetch_add(1, Ordering::Release);
    }
}

#[inline(never)]
#[track_caller]
#[cold]
const fn too_many_refs() -> ! {
    panic!("Too many `Ref` instances created");
}
