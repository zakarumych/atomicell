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

/// Lock type used by [`AtomicCell`].
pub type Lock = AtomicIsize;

/// Create atomic borrow lock.
/// Initially not borrowed.
pub const fn new_lock() -> Lock {
    Lock::new(0)
}

/// Limit on the count of concurrent [`Ref`] instances for the same [`AtomicCell`].
const REF_LIMIT_FLAG: isize = 1 + (isize::MAX >> 1);

/// Dummy lock value that is used to create [`Ref`] and [`RefMut`] from references instead of [`AtomicCell`].
pub(crate) static DUMMY_LOCK: Lock = new_lock();

fn is_dummy(lock: &Lock) -> bool {
    core::ptr::eq(lock, &DUMMY_LOCK)
}

/// Returns true if specified lock value is borrowed immutably.
#[inline(always)]
pub fn is_reading(v: isize) -> bool {
    v > 0
}

/// Returns true if specified lock value is borrowed mutably.
#[inline(always)]
pub fn is_writing(v: isize) -> bool {
    v < 0
}

/// Returns true if specified lock value is borrowed.
#[inline(always)]
pub fn is_borrowed(v: isize) -> bool {
    v != 0
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

/// Attempts to borrow specified lock immutably.
#[inline]
pub fn try_borrow(lock: &Lock) -> bool {
    debug_assert!(!is_dummy(lock), "dummy lock cannot be used from outside");

    loop {
        // Get original value.
        let val = lock.load(Ordering::Relaxed);

        if is_writing(val) {
            // Locked for writing.
            return false;
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

        return ok;
    }
}

/// Clones immutable borrow of specified lock.
/// This function MUST be called only when lock is already borrowed immutably.
///
/// # Safety
///
/// This function is safe but must be used with care to ensure locking correctness.
#[inline]
pub fn clone_borrow(lock: &Lock) {
    if is_dummy(lock) {
        // Dummy lock is always borrowed both mutably and immutably.
        // That lock is never used to sync access to any value.
        return;
    }

    // Lock is already held, so `Relaxed` ordering is fine.
    let old = lock.fetch_add(1, Ordering::Relaxed);

    // Check that counter won't overflow into writing state.
    // REF_LIMIT_FLAG allows plenty clones to be created.
    // In fact, without forgetting to release borrowed lock, the counter cannot overflow
    // And there can be `REF_LIMIT_FLAG - 1` attempts to create excessive borrows
    // that assumed to never happen as there can't be that much threads.
    if check_read_refs_count(old) {
        // Forbid locking this much times at once.
        // Balance lock increment with decrement.
        lock.fetch_sub(1, Ordering::Relaxed);

        // Panic is put into separate non-inlineable cold function
        // in order to not pollute this function
        // and hint compiler that this branch is unlikely.
        too_many_refs();
    }
}

/// Releases immutable borrow of specified lock.
/// This function MUST be called only when lock is borrowed immutably.
/// This function MUST be called only once for each succefful borrow and borrow clone.
#[inline]
pub fn release_borrow(lock: &Lock) {
    if is_dummy(lock) {
        return;
    }

    debug_assert!(is_dummy(lock) || is_reading(lock.load(Ordering::Relaxed)));

    // Reset the lock counter. `Release` semantics is required
    // to sync with `Acquire` semantics within `try_borrow` and `try_borrow_mut`.
    lock.fetch_sub(1, Ordering::Release);
}

/// Attempts to borrow specified lock mutably.
#[inline]
pub fn try_borrow_mut(lock: &Lock) -> bool {
    debug_assert!(!is_dummy(lock), "dummy lock cannot be used from outside");

    // `Acquire` semantics syncs this operation with `Release` semantics in `Ref::drop` and `RefMut::drop`.
    // Lock counter `0` ensures that no other borrows exist.
    let ok = lock
        .compare_exchange(0, -1, Ordering::Acquire, Ordering::Relaxed)
        .is_ok();

    ok
}

/// Clones mutable borrow of specified lock.
/// This function MUST be called only when lock is already borrowed mutably.
///
/// # Safety
///
/// This function is safe but must be used with care to ensure locking correctness.
#[inline]
pub fn clone_borrow_mut(lock: &Lock) {
    if is_dummy(lock) {
        // Dummy lock is always borrowed both mutably and immutably.
        // That lock is never used to sync access to any value.
        return;
    }

    // Lock is already held by `RefMut` instance,
    // so `Relaxed` ordering is fine.
    let old = lock.fetch_sub(1, Ordering::Relaxed);

    // Check that counter won't overflow into reading state.
    // REF_LIMIT_FLAG allows plenty clones to be created.
    // In fact, without `forget`ing the `AtomicBorrowMut` instances, the counter cannot overflow
    // because this much `AtomicBorrowMut` instances won't fit into address space.
    // And there can be `REF_LIMIT_FLAG - 1` attempts to create excessive `AtomicBorrowMut`s
    // that assumed to never happen as there can't be that much threads.
    if check_write_refs_count(old) {
        // Forbid creating this much `AtomicBorrowMut` instances.
        // Balance lock decrement with increment.
        lock.fetch_add(1, Ordering::Relaxed);

        // Panic is put into separate non-inlineable cold function
        // in order to not pollute this function
        // and hit compiler that this branch is unlikely.
        too_many_refs();
    }
}

/// Releases mutable borrow of specified lock.
/// This function MUST be called only when lock is borrowed mutably.
/// This function MUST be called only once for each succefful borrow and borrow clone.
#[inline]
pub fn release_borrow_mut(lock: &Lock) {
    if is_dummy(lock) {
        return;
    }

    debug_assert!(is_dummy(lock) || is_writing(lock.load(Ordering::Relaxed)));

    // Increment lock counter. `Release` semantics is required
    // to sync with `Acquire` semantics within `try_borrow` and `try_borrow_mut`.
    lock.fetch_add(1, Ordering::Release);
}

#[inline(never)]
#[track_caller]
#[cold]
const fn too_many_refs() -> ! {
    panic!("Too many `Ref` instances created");
}

/// Encapsulates shared borrowing state.
///
/// An instance of this type guarantees that [`AtomicBorrowMut`] cannot be constructed for the same lock.
#[repr(transparent)]
pub struct AtomicBorrow<'a> {
    lock: &'a Lock,
}

impl<'a> AtomicBorrow<'a> {
    /// Attempts to borrow lock immutably.
    ///
    /// Fails if `Lock` contains a value for which `is_writing` returns true.
    ///
    /// On success `Lock` contains value for which `is_reading` returns true.
    #[inline(always)]
    pub fn try_new(lock: &'a Lock) -> Option<Self> {
        if try_borrow(lock) {
            // Lock is borrowed.
            Some(AtomicBorrow { lock })
        } else {
            None
        }
    }

    /// Restore previously leaked [`AtomicBorrow`] instance.
    ///
    /// This method should be called only after forgetting [`AtomicBorrow`] instance.
    /// Or after locking manually using [`try_borrow`] method.
    #[inline(always)]
    pub unsafe fn restore_leaked(lock: &'a Lock) -> Self {
        AtomicBorrow { lock }
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
    #[inline(always)]
    pub fn clone(&self) -> AtomicBorrow<'a> {
        clone_borrow(self.lock);
        AtomicBorrow { lock: self.lock }
    }
}

impl<'a> Drop for AtomicBorrow<'a> {
    #[inline(always)]
    fn drop(&mut self) {
        release_borrow(self.lock)
    }
}

/// Encapsulates exclusive borrowing state.
///
/// An instance of this type guarantees that neither [`AtomicBorrowMut`], nor [`AtomicBorrow`] cannot be constructed for the same lock.
#[repr(transparent)]
pub struct AtomicBorrowMut<'a> {
    lock: &'a Lock,
}

impl<'a> AtomicBorrowMut<'a> {
    /// Attempts to borrow lock mutably.
    #[inline(always)]
    pub fn try_new(lock: &'a Lock) -> Option<Self> {
        if try_borrow_mut(lock) {
            Some(AtomicBorrowMut { lock })
        } else {
            None
        }
    }

    /// Restore previously leaked [`AtomicBorrowMut`] instance.
    ///
    /// This method should be called only after forgetting [`AtomicBorrowMut`] instance.
    /// Or after locking manually using [`try_borrow_mut`] method.
    #[inline(always)]
    pub unsafe fn restore_leaked(lock: &'a Lock) -> Self {
        AtomicBorrowMut { lock }
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
    #[inline(always)]
    pub fn clone(&self) -> AtomicBorrowMut<'a> {
        clone_borrow_mut(self.lock);
        AtomicBorrowMut { lock: self.lock }
    }
}

impl<'a> Drop for AtomicBorrowMut<'a> {
    #[inline(always)]
    fn drop(&mut self) {
        release_borrow_mut(self.lock);
    }
}
