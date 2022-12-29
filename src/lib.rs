//! Complete implementation of atomic cell.
//!
//! [`AtomicCell`] behaves like a [`AtomicCell`] but with atomic operations
//! that allows sharing it across threads if stored value is [`Send`] and [`Sync`].
//!
//! [`AtomicCell`]: https://doc.rust-lang.org/core/cell/struct.AtomicCell.html

#![no_std]

use core::{
    borrow::{Borrow, BorrowMut},
    cell::UnsafeCell,
    cmp,
    convert::{AsMut, AsRef},
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut, RangeBounds},
    sync::atomic::{AtomicIsize, Ordering},
};

/// Limit on the count of concurrent [`Ref`] instances for the same [`AtomicCell`].
const REF_LIMIT_FLAG: isize = 1 + (isize::MAX >> 1);

/// Dummy lock value that is used to create [`Ref`] and [`RefMut`] from references instead of [`AtomicCell`].
static DUMMY_LOCK: AtomicIsize = AtomicIsize::new(0);

fn is_dummy(lock: &AtomicIsize) -> bool {
    core::ptr::eq(lock, &DUMMY_LOCK)
}

/// Returns true if lock is locked for reads or writes.
#[inline(always)]
fn is_reading(v: isize) -> bool {
    v > 0
}

/// Returns true if lock is locked for writes.
#[inline(always)]
fn is_writing(v: isize) -> bool {
    v < 0
}

/// Encapsulates shared borrowing of the [`AtomicCell`].
/// An instance of this type guarantees that `AtomicBorrowMut` cannot be constructed for the same lock.
struct AtomicBorrow<'a> {
    lock: &'a AtomicIsize,
}

impl<'a> AtomicBorrow<'a> {
    /// Attempts to borrow lock immutably.
    #[inline]
    fn try_new(lock: &'a AtomicIsize) -> Option<Self> {
        loop {
            // Get original value.
            let val = lock.load(Ordering::Relaxed);

            // This cannot overflow because the value is kept below `REF_LIMIT_FLAG`.
            if !is_reading(val + 1) {
                // Locked for writing.
                return None;
            }

            // Check that counter won't overflow into writing state.
            // `REF_LIMIT_FLAG` allows plenty clones to be created.
            // In fact, without `forget`ing the `Ref` instances, the counter cannot overflow
            // because this much `Ref` instances won't fit into address space.
            // `REF_LIMIT_FLAG` value allows `REF_LIMIT_FLAG - 1` concurrent attempts to borrow immutably.
            // Which assumed to never happen as there can't be that much threads.
            if val & REF_LIMIT_FLAG != 0 {
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
    /// It is used within [`Ref::new`] and [`RefMut::new`] methods that wrap take external references.
    #[inline(always)]
    fn dummy() -> Self {
        AtomicBorrow { lock: &DUMMY_LOCK }
    }

    /// Shared borrows can be cloned.
    /// There is a hard limit on number of clones for each lock (except dummy lock) and it will panic if cloning fails.
    /// Huge limit value makes it practically unreachable without `forget`ing clones in a loop.
    #[inline]
    fn clone(&self) -> AtomicBorrow<'a> {
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
        if old & REF_LIMIT_FLAG == REF_LIMIT_FLAG {
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
        debug_assert!(is_dummy(self.lock) || is_reading(self.lock.load(Ordering::Relaxed)));

        // Reset the lock counter. `Release` semantics is required
        // to sync with `Acquire` semantics within `try_borrow` and `try_borrow_mut`.
        self.lock.fetch_sub(1, Ordering::Release);
    }
}

/// Encapsulates exclusive borrowing of the `AtomicCell`.
/// An instance of this type guarantees that `AtomicBorrow` cannot be constructed for the same lock.
struct AtomicBorrowMut<'a> {
    lock: &'a AtomicIsize,
}

impl<'a> AtomicBorrowMut<'a> {
    /// Attempts to borrow lock mutably.
    #[inline]
    fn try_new(lock: &'a AtomicIsize) -> Option<Self> {
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
    /// It is paired with `Ref` and `RefMut` constructors that take external references.
    #[inline(always)]
    fn dummy() -> Self {
        AtomicBorrowMut { lock: &DUMMY_LOCK }
    }

    /// Shared borrows can be cloned.
    /// There is a hard limit on number of clones for each lock (except dummy lock) and it will panic if cloning fails.
    /// Huge limit value makes it practically unreachable without `forget`ing clones in a loop.
    #[inline]
    fn clone(&self) -> AtomicBorrowMut<'a> {
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
        if old & REF_LIMIT_FLAG == 0 {
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
        debug_assert!(is_dummy(self.lock) || is_writing(self.lock.load(Ordering::Relaxed)));

        // Increment lock counter. `Release` semantics is required
        // to sync with `Acquire` semantics within `try_borrow` and `try_borrow_mut`.
        self.lock.fetch_add(1, Ordering::Release);
    }
}

/// Wrapper for a borrowed [`AtomicCell`] that will released lock on drop.
///
/// This type can be dereferenced to [`&T`].
///
/// Implements [`Borrow<T>`] and [`AsRef<T>`] for convenience.
///
/// Implements [`Debug`], [`Display`], [`PartialEq`], [`PartialOrd`] and [`Hash`] by delegating to `T`.
///
/// [`&T`]: https://doc.rust-lang.org/core/primitive.reference.html
pub struct Ref<'a, T: ?Sized> {
    value: &'a T,
    borrow: AtomicBorrow<'a>,
}

impl<'a, T> Clone for Ref<'a, T>
where
    T: ?Sized,
{
    #[inline]
    fn clone(&self) -> Self {
        Ref {
            borrow: self.borrow.clone(),
            value: self.value,
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        self.borrow = source.borrow.clone();
        self.value = source.value;
    }
}

impl<'a, T> Deref for Ref<'a, T>
where
    T: ?Sized,
{
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        self.value
    }
}

impl<'a, T> Debug for Ref<'a, T>
where
    T: Debug,
{
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(self.value, f)
    }
}

impl<'a, T> Display for Ref<'a, T>
where
    T: Display,
{
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(self.value, f)
    }
}

impl<'a, T, U> PartialEq<U> for Ref<'a, T>
where
    T: PartialEq<U>,
{
    #[inline(always)]
    fn eq(&self, other: &U) -> bool {
        PartialEq::eq(self.value, other)
    }
}

impl<'a, T, U> PartialOrd<U> for Ref<'a, T>
where
    T: PartialOrd<U>,
{
    #[inline(always)]
    fn partial_cmp(&self, other: &U) -> Option<cmp::Ordering> {
        PartialOrd::partial_cmp(self.value, other)
    }
}

impl<'a, T> Hash for Ref<'a, T>
where
    T: Hash,
{
    #[inline(always)]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        Hash::hash(self.value, state)
    }
}

impl<'a, T> Borrow<T> for Ref<'a, T> {
    #[inline(always)]
    fn borrow(&self) -> &T {
        self.value
    }
}

impl<'a, T, U> AsRef<U> for Ref<'a, T>
where
    T: AsRef<U>,
{
    #[inline(always)]
    fn as_ref(&self) -> &U {
        self.value.as_ref()
    }
}

impl<'a, T> Ref<'a, T>
where
    T: ?Sized,
{
    /// Wraps external reference into [`Ref`].
    ///
    /// This function's purpose is to satisfy type requirements
    /// where [`Ref`] is required but reference does not live in [`AtomicCell`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::Ref;
    ///
    /// let r = Ref::new(&42);
    /// ```
    #[inline]
    pub fn new(r: &'a T) -> Self {
        Ref {
            value: r,
            borrow: AtomicBorrow::dummy(),
        }
    }

    /// Makes a new [`Ref`] for a component of the borrowed data.
    ///
    /// The [`AtomicCell`] is already immutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as Ref::map(...).
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, Ref};
    ///
    /// let c = AtomicCell::new((5, 'b'));
    /// let b1: Ref<(u32, char)> = c.borrow();
    /// let b2: Ref<u32> = Ref::map(b1, |t| &t.0);
    /// assert_eq!(*b2, 5)
    /// ```
    #[inline]
    pub fn map<F, U>(r: Ref<'a, T>, f: F) -> Ref<'a, U>
    where
        F: FnOnce(&T) -> &U,
        U: ?Sized,
    {
        Ref {
            value: f(r.value),
            borrow: r.borrow,
        }
    }

    /// Makes a new [`Ref`] for an optional component of the borrowed data.
    /// The original guard is returned as an Err(..) if the closure returns None.
    ///
    /// The [`AtomicCell`] is already mutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as Ref::filter_map(...).
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, Ref};
    /// let c = AtomicCell::new(vec![1, 2, 3]);
    /// let b1: Ref<Vec<u32>> = c.borrow();
    /// let b2: Result<Ref<u32>, _> = Ref::filter_map(b1, |v| v.get(1));
    /// assert_eq!(*b2.unwrap(), 2);
    /// ```
    pub fn filter_map<U: ?Sized, F>(r: Ref<'a, T>, f: F) -> Result<Ref<'a, U>, Self>
    where
        F: FnOnce(&T) -> Option<&U>,
    {
        match f(r.value) {
            Some(value) => Ok(Ref {
                value,
                borrow: r.borrow,
            }),
            None => Err(r),
        }
    }

    /// Splits a [`Ref`] into multiple [`Ref`]s for different components of the borrowed data.
    ///
    /// The [`AtomicCell`] is already immutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as `Ref::map_split(...)`.
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{Ref, AtomicCell};
    ///
    /// let cell = AtomicCell::new([1, 2, 3, 4]);
    /// let borrow = cell.borrow();
    /// let (begin, end) = Ref::map_split(borrow, |slice| slice.split_at(2));
    /// assert_eq!(*begin, [1, 2]);
    /// assert_eq!(*end, [3, 4]);
    /// ```
    pub fn map_split<U, V, F>(r: Ref<'a, T>, f: F) -> (Ref<'a, U>, Ref<'a, V>)
    where
        U: ?Sized,
        V: ?Sized,
        F: FnOnce(&T) -> (&U, &V),
    {
        let borrow_u = r.borrow.clone();
        let borrow_v = r.borrow;

        let (u, v) = f(r.value);

        (
            Ref {
                value: u,
                borrow: borrow_u,
            },
            Ref {
                value: v,
                borrow: borrow_v,
            },
        )
    }

    /// Convert into a reference to the underlying data.
    ///
    /// The underlying [`AtomicCell`] can never be mutably borrowed from again
    /// and will always appear already immutably borrowed.
    /// It is not a good idea to leak more than a constant number of references.
    /// The [`AtomicCell`] can be immutably borrowed again if only a smaller number of leaks have occurred in total.
    ///
    /// This is an associated function that needs to be used as Ref::leak(...).
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, Ref};
    /// let cell = AtomicCell::new(0);
    ///
    /// let value = Ref::leak(cell.borrow());
    /// assert_eq!(*value, 0);
    ///
    /// assert!(cell.try_borrow().is_some());
    /// assert!(cell.try_borrow_mut().is_none());
    /// ```
    pub fn leak(r: Ref<'a, T>) -> &'a T {
        core::mem::forget(r.borrow);
        r.value
    }

    /// Converts reference and returns result wrapped in the [`Ref`].
    ///
    /// The [`AtomicCell`] is already immutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as `Ref::map_split(...)`.
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, Ref};
    ///
    /// let c = AtomicCell::new(String::from("hello"));
    /// let b1: Ref<String> = c.borrow();
    /// let b2: Ref<str> = Ref::as_ref(b1);
    /// assert_eq!(*b2, *"hello")
    /// ```
    #[inline]
    pub fn as_ref<U>(r: Ref<'a, T>) -> Ref<'a, U>
    where
        U: ?Sized,
        T: AsRef<U>,
    {
        Ref {
            value: r.value.as_ref(),
            borrow: r.borrow,
        }
    }

    /// Dereferences and returns result wrapped in the [`Ref`].
    ///
    /// The [`AtomicCell`] is already immutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as `Ref::map_split(...)`.
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, Ref};
    ///
    /// let c = AtomicCell::new(String::from("hello"));
    /// let b1: Ref<String> = c.borrow();
    /// let b2: Ref<str> = Ref::as_deref(b1);
    /// assert_eq!(*b2, *"hello")
    /// ```
    #[inline]
    pub fn as_deref(r: Ref<'a, T>) -> Ref<'a, T::Target>
    where
        T: Deref,
    {
        Ref {
            value: &r.value,
            borrow: r.borrow,
        }
    }
}

impl<'a, T> Ref<'a, Option<T>> {
    /// Transposes a [`Ref`] of an [`Option`] into an [`Option`] of a [`Ref`].
    /// Releases shared lock of [`AtomicCell`] if the value is [`None`].
    ///
    /// The [`AtomicCell`] is already immutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as `Ref::map_split(...)`.
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, Ref};
    ///
    /// let c = AtomicCell::new(Some(5));
    /// let b1: Ref<Option<i32>> = c.borrow();
    /// let b2: Option<Ref<i32>> = Ref::transpose(b1);
    /// assert!(b2.is_some());
    ///
    /// let c = AtomicCell::new(None);
    /// let b1: Ref<Option<i32>> = c.borrow();
    /// let b2: Option<Ref<i32>> = Ref::transpose(b1);
    /// assert!(b2.is_none());
    /// assert!(c.try_borrow_mut().is_some());
    /// ```
    #[inline]
    pub fn transpose(r: Ref<'a, Option<T>>) -> Option<Ref<'a, T>> {
        Some(Ref {
            value: r.value.as_ref()?,
            borrow: r.borrow,
        })
    }
}

impl<'a, T> Ref<'a, [T]> {
    /// Makes a new [`Ref`] for a sub-slice of the borrowed slice.
    ///
    /// The [`AtomicCell`] is already immutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as Ref::map(...).
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, Ref};
    ///
    /// let c = AtomicCell::new([1, 2, 3, 4, 5]);
    /// let b1: Ref<[i32]> = Ref::as_ref(c.borrow());
    /// let b2: Ref<[i32]> = Ref::slice(b1, 2..4);
    /// assert_eq!(*b2, [3, 4])
    /// ```
    #[inline]
    pub fn slice<R>(r: Ref<'a, [T]>, range: R) -> Ref<'a, [T]>
    where
        R: RangeBounds<usize>,
    {
        let bounds = (range.start_bound().cloned(), range.end_bound().cloned());
        Ref {
            value: &r.value[bounds],
            borrow: r.borrow,
        }
    }
}

/// Wrapper for mutably borrowed [`AtomicCell`] that will released lock on drop.
///
/// This type can be dereferenced to [`&mut T`].
///
/// Implements [`Borrow<T>`], [`BorrowMut<T>`], [`AsRef<T>`] and [`AsMut<T>`] for convenience.
///
/// Implements [`Debug`], [`Display`], [`PartialEq`], [`PartialOrd`] and [`Hash`] by delegating to `T`.
///
/// [`&T`]: https://doc.rust-lang.org/core/primitive.reference.html
pub struct RefMut<'a, T: ?Sized> {
    value: &'a mut T,
    borrow: AtomicBorrowMut<'a>,
}

impl<'a, T> Deref for RefMut<'a, T>
where
    T: ?Sized,
{
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<'a, T> DerefMut for RefMut<'a, T>
where
    T: ?Sized,
{
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<'a, T> Debug for RefMut<'a, T>
where
    T: Debug,
{
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(self.value, f)
    }
}

impl<'a, T> Display for RefMut<'a, T>
where
    T: Display,
{
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(self.value, f)
    }
}

impl<'a, T, U> PartialEq<U> for RefMut<'a, T>
where
    T: PartialEq<U>,
{
    #[inline(always)]
    fn eq(&self, other: &U) -> bool {
        PartialEq::eq(&*self.value, other)
    }
}

impl<'a, T, U> PartialOrd<U> for RefMut<'a, T>
where
    T: PartialOrd<U>,
{
    #[inline(always)]
    fn partial_cmp(&self, other: &U) -> Option<cmp::Ordering> {
        PartialOrd::partial_cmp(&*self.value, other)
    }
}

impl<'a, T> Hash for RefMut<'a, T>
where
    T: Hash,
{
    #[inline(always)]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        Hash::hash(self.value, state)
    }
}

impl<'a, T> Borrow<T> for RefMut<'a, T> {
    #[inline(always)]
    fn borrow(&self) -> &T {
        &self.value
    }
}

impl<'a, T> BorrowMut<T> for RefMut<'a, T> {
    #[inline(always)]
    fn borrow_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<'a, T, U> AsRef<U> for RefMut<'a, T>
where
    T: AsRef<U>,
{
    #[inline(always)]
    fn as_ref(&self) -> &U {
        self.value.as_ref()
    }
}

impl<'a, T, U> AsMut<U> for RefMut<'a, T>
where
    T: AsMut<U>,
{
    #[inline(always)]
    fn as_mut(&mut self) -> &mut U {
        self.value.as_mut()
    }
}

impl<'a, T> RefMut<'a, T>
where
    T: ?Sized,
{
    /// Wraps external reference into [`RefMut`].
    ///
    /// This function's purpose is to satisfy type requirements
    /// where [`RefMut`] is required but reference does not live in [`AtomicCell`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::RefMut;
    ///
    /// let mut value = 42;
    /// let r = RefMut::new(&mut value);
    #[inline]
    pub fn new(r: &'a mut T) -> Self {
        RefMut {
            value: r,
            borrow: AtomicBorrowMut::dummy(),
        }
    }

    /// Makes a new [`RefMut`] for a component of the borrowed data.
    ///
    /// The [`AtomicCell`] is already mutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as RefMut::map(...).
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, RefMut};
    ///
    /// let c = AtomicCell::new((5, 'b'));
    /// let b1: RefMut<(u32, char)> = c.borrow_mut();
    /// let b2: RefMut<u32> = RefMut::map(b1, |t| &mut t.0);
    /// assert_eq!(*b2, 5)
    #[inline]
    pub fn map<F, U>(r: RefMut<'a, T>, f: F) -> RefMut<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
        U: ?Sized,
    {
        RefMut {
            value: f(r.value),
            borrow: r.borrow,
        }
    }

    /// Makes a new [`RefMut`] for an optional component of the borrowed data.
    /// The original guard is returned as an Err(..) if the closure returns None.
    ///
    /// The [`AtomicCell`] is already mutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as RefMut::filter_map(...).
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, RefMut};
    ///
    /// let c = AtomicCell::new(vec![1, 2, 3]);
    ///
    /// {
    ///     let b1: RefMut<Vec<u32>> = c.borrow_mut();
    ///     let mut b2: Result<RefMut<u32>, _> = RefMut::filter_map(b1, |v| v.get_mut(1));
    ///
    ///     if let Ok(mut b2) = b2 {
    ///         *b2 += 2;
    ///     }
    /// }
    ///
    /// assert_eq!(*c.borrow(), vec![1, 4, 3]);
    /// ```
    pub fn filter_map<U: ?Sized, F>(r: RefMut<'a, T>, f: F) -> Result<RefMut<'a, U>, Self>
    where
        F: FnOnce(&mut T) -> Option<&mut U>,
    {
        // FIXME(nll-rfc#40): fix borrow-check
        let RefMut { value, borrow } = r;
        let value = value as *mut T;
        // SAFETY: function holds onto an exclusive reference for the duration
        // of its call through `r`, and the pointer is only de-referenced
        // inside of the function call never allowing the exclusive reference to
        // escape.
        match f(unsafe { &mut *value }) {
            Some(value) => Ok(RefMut { value, borrow }),
            None => {
                // SAFETY: same as above.
                Err(RefMut {
                    value: unsafe { &mut *value },
                    borrow,
                })
            }
        }
    }

    /// Splits a [`RefMut`] into multiple [`RefMut`]s for different components of the borrowed data.
    ///
    /// The [`AtomicCell`] is already immutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as `RefMut::map_split(...)`.
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{RefMut, AtomicCell};
    ///
    /// let cell = AtomicCell::new([1, 2, 3, 4]);
    /// let borrow = cell.borrow_mut();
    /// let (begin, end) = RefMut::map_split(borrow, |slice| slice.split_at_mut(2));
    /// assert_eq!(*begin, [1, 2]);
    /// assert_eq!(*end, [3, 4]);
    /// ```
    pub fn map_split<U, V, F>(r: RefMut<'a, T>, f: F) -> (RefMut<'a, U>, RefMut<'a, V>)
    where
        U: ?Sized,
        V: ?Sized,
        F: FnOnce(&mut T) -> (&mut U, &mut V),
    {
        let borrow_u = r.borrow.clone();
        let borrow_v = r.borrow;

        let (u, v) = f(r.value);

        (
            RefMut {
                value: u,
                borrow: borrow_u,
            },
            RefMut {
                value: v,
                borrow: borrow_v,
            },
        )
    }

    /// Convert into a reference to the underlying data.
    ///
    /// The underlying [`AtomicCell`] can never be mutably borrowed from again
    /// and will always appear already immutably borrowed.
    /// It is not a good idea to leak more than a constant number of references.
    /// The [`AtomicCell`] can be immutably borrowed again if only a smaller number of leaks have occurred in total.
    ///
    /// This is an associated function that needs to be used as RefMut::leak(...).
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, RefMut};
    /// let cell = AtomicCell::new(0);
    ///
    /// let value = RefMut::leak(cell.borrow_mut());
    /// assert_eq!(*value, 0);
    /// *value = 1;
    /// assert_eq!(*value, 1);
    ///
    /// assert!(cell.try_borrow().is_none());
    /// assert!(cell.try_borrow_mut().is_none());
    /// ```
    pub fn leak(r: RefMut<'a, T>) -> &'a mut T {
        core::mem::forget(r.borrow);
        r.value
    }

    /// Converts reference and returns result wrapped in the [`RefMut`].
    ///
    /// The [`AtomicCell`] is already mutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as `RefMut::map_split(...)`.
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, RefMut};
    ///
    /// let c = AtomicCell::new(String::from("hello"));
    /// let b1: RefMut<String> = c.borrow_mut();
    /// let mut b2: RefMut<str> = RefMut::as_mut(b1);
    /// b2.make_ascii_uppercase();
    /// assert_eq!(*b2, *"HELLO")
    /// ```
    #[inline]
    pub fn as_mut<U>(r: RefMut<'a, T>) -> RefMut<'a, U>
    where
        U: ?Sized,
        T: AsMut<U>,
    {
        RefMut {
            value: r.value.as_mut(),
            borrow: r.borrow,
        }
    }

    /// Dereferences and returns result wrapped in the [`RefMut`].
    ///
    /// The [`AtomicCell`] is already mutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as `RefMut::map_split(...)`.
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, RefMut};
    ///
    /// let c = AtomicCell::new(String::from("hello"));
    /// let b1: RefMut<String> = c.borrow_mut();
    /// let mut b2: RefMut<str> = RefMut::as_deref_mut(b1);
    /// b2.make_ascii_uppercase();
    /// assert_eq!(*b2, *"HELLO")
    /// ```
    #[inline]
    pub fn as_deref_mut(r: RefMut<'a, T>) -> RefMut<'a, T::Target>
    where
        T: DerefMut,
    {
        RefMut {
            value: &mut *r.value,
            borrow: r.borrow,
        }
    }
}

impl<'a, T> RefMut<'a, Option<T>> {
    /// Transposes a [`RefMut`] of an [`Option`] into an [`Option`] of a [`RefMut`].
    /// Releases shared lock of [`AtomicCell`] if the value is [`None`].
    ///
    /// The [`AtomicCell`] is already mutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as `RefMut::map_split(...)`.
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, RefMut};
    ///
    /// let c = AtomicCell::new(Some(5));
    /// let b1: RefMut<Option<i32>> = c.borrow_mut();
    /// let b2: Option<RefMut<i32>> = RefMut::transpose(b1);
    /// assert!(b2.is_some());
    ///
    /// let c = AtomicCell::new(None);
    /// let b1: RefMut<Option<i32>> = c.borrow_mut();
    /// let b2: Option<RefMut<i32>> = RefMut::transpose(b1);
    /// assert!(b2.is_none());
    /// assert!(c.try_borrow_mut().is_some());
    /// ```
    #[inline]
    pub fn transpose(r: RefMut<'a, Option<T>>) -> Option<RefMut<'a, T>> {
        Some(RefMut {
            value: r.value.as_mut()?,
            borrow: r.borrow,
        })
    }
}

impl<'a, T> RefMut<'a, [T]> {
    /// Makes a new [`RefMut`] for a sub-slice of the borrowed slice.
    ///
    /// The [`AtomicCell`] is already mutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as RefMut::map(...).
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, RefMut};
    ///
    /// let c = AtomicCell::new([1, 2, 3, 4, 5]);
    /// let b1: RefMut<[i32]> = RefMut::as_mut(c.borrow_mut());
    /// let b2: RefMut<[i32]> = RefMut::slice(b1, 2..4);
    /// assert_eq!(*b2, [3, 4])
    /// ```
    #[inline]
    pub fn slice<R>(r: RefMut<'a, [T]>, range: R) -> RefMut<'a, [T]>
    where
        R: RangeBounds<usize>,
    {
        let bounds = (range.start_bound().cloned(), range.end_bound().cloned());
        RefMut {
            value: &mut r.value[bounds],
            borrow: r.borrow,
        }
    }
}

/// A mutable memory location with dynamically checked borrow rules
/// This type is similar to [`core::cell::AtomicCell`].
/// The main difference is that this type uses atomic operations for borrowing.
/// Thus allowing to use it in multi-threaded environment.
///
/// `AtomicCell<T>` implements `Send` if `T: Send`.
/// `AtomicCell<T>` implements `Sync` if `T: Send + Sync`.
pub struct AtomicCell<T: ?Sized> {
    lock: AtomicIsize,
    value: UnsafeCell<T>,
}

/// `AtomicCell` can be sent to another thread if value can be sent.
/// Sending can occur on owned cell or mutable reference to it.
/// Either way it is not borrowed, so it is impossible to share stored value this way.
unsafe impl<T> Send for AtomicCell<T> where T: Send {}

/// `AtomicCell` can be shared across threads if value can be sent and shared.
/// Requires `T: Send` because mutable borrow can occur in another thread.
/// Requires `T: Sync` because immutable borrows could occur concurrently in different threads.
unsafe impl<T> Sync for AtomicCell<T> where T: Send + Sync {}

impl<T> AtomicCell<T> {
    /// Creates a new [`AtomicCell`] containing value.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::AtomicCell;
    ///
    /// let cell = AtomicCell::new(5);
    /// ```
    #[inline]
    pub const fn new(value: T) -> Self {
        AtomicCell {
            value: UnsafeCell::new(value),
            lock: AtomicIsize::new(0),
        }
    }

    /// Consumes the [`AtomicCell`], returning the wrapped value.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::AtomicCell;
    ///
    /// let c = AtomicCell::new(5);
    ///
    /// let five = c.into_inner();
    /// ```
    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.value.into_inner()
    }

    /// Replaces the wrapped value with a new one, returning the old value, without deinitializing either one.
    /// This function corresponds to [core::mem::replace].
    ///
    /// # Panics
    ///
    /// Panics if the value is currently borrowed.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::AtomicCell;
    /// let cell = AtomicCell::new(5);
    /// let old_value = cell.replace(6);
    /// assert_eq!(old_value, 5);
    /// assert_eq!(cell, AtomicCell::new(6));
    /// ```
    #[inline(always)]
    pub fn replace(&self, t: T) -> T {
        core::mem::replace(&mut *self.borrow_mut(), t)
    }

    /// Replaces the wrapped value with a new one computed from f, returning the old value, without deinitializing either one.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently borrowed.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::AtomicCell;
    /// let cell = AtomicCell::new(5);
    /// let old_value = cell.replace_with(|&mut old| old + 1);
    /// assert_eq!(old_value, 5);
    /// assert_eq!(cell, AtomicCell::new(6));
    /// ```
    #[inline]
    pub fn replace_with<F: FnOnce(&mut T) -> T>(&self, f: F) -> T {
        match self.try_borrow_mut() {
            None => failed_to_borrow_mut(),
            Some(mut borrow) => {
                let t = f(&mut borrow.value);
                core::mem::replace(&mut borrow.value, t)
            }
        }
    }

    /// Swaps the wrapped value of self with the wrapped value of other, without deinitializing either one.
    /// This function corresponds to [core::mem::swap].
    ///
    /// # Panics
    ///
    /// Panics if the value in either AtomicCell is currently borrowed.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::AtomicCell;
    /// let c = AtomicCell::new(5);
    /// let d = AtomicCell::new(6);
    /// c.swap(&d);
    /// assert_eq!(c, AtomicCell::new(6));
    /// assert_eq!(d, AtomicCell::new(5));
    /// ```
    #[inline]
    pub fn swap(&self, other: &Self) {
        match self.try_borrow_mut() {
            None => failed_to_borrow_mut(),
            Some(borrow) => match other.try_borrow_mut() {
                None => failed_to_borrow_mut(),
                Some(other_borrow) => {
                    core::mem::swap(&mut *borrow.value, &mut *other_borrow.value);
                }
            },
        }
    }
}

impl<T> AtomicCell<T>
where
    T: ?Sized,
{
    /// Immutably borrows the wrapped value, returning [`None`] if the value is currently mutably borrowed.
    ///
    /// The borrow lasts until the returned [`Ref`] exits scope. Multiple immutable borrows can be taken out at the same time.
    ///
    /// This is the non-panicking variant of borrow.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::AtomicCell;
    /// let c = AtomicCell::new(5);
    ///
    /// {
    ///     let m = c.borrow_mut();
    ///     assert!(c.try_borrow().is_none());
    /// }
    ///
    /// {
    ///     let m = c.borrow();
    ///     assert!(c.try_borrow().is_some());
    /// }
    /// ```
    #[inline]
    pub fn try_borrow(&self) -> Option<Ref<'_, T>> {
        // Acquire shared borrow.
        match AtomicBorrow::try_new(&self.lock) {
            None => None,
            Some(borrow) => {
                // It is now safe to construct immutable borrow.
                Some(Ref {
                    borrow,
                    value: unsafe {
                        // Locking mechanism ensures that mutable aliasing is impossible.
                        &*self.value.get()
                    },
                })
            }
        }
    }

    /// Immutably borrows the wrapped value.
    ///
    /// The borrow lasts until the returned [`Ref`] exits scope. Multiple immutable borrows can be taken out at the same time.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently mutably borrowed. For a non-panicking variant, use [`try_borrow`].
    ///
    /// # Examples
    ///
    /// [`try_borrow`]: #method.try_borrow
    ///
    /// ```
    /// use atomicell::AtomicCell;
    ///
    /// let c = AtomicCell::new(5);
    ///
    /// let borrowed_five = c.borrow();
    /// let borrowed_five2 = c.borrow();
    /// ```
    ///
    /// An example of panic:
    ///
    /// ```should_panic
    /// use atomicell::AtomicCell;
    ///
    /// let c = AtomicCell::new(5);
    ///
    /// let m = c.borrow_mut();
    /// let b = c.borrow(); // this causes a panic
    /// ```
    #[inline(always)]
    #[track_caller]
    pub fn borrow(&self) -> Ref<'_, T> {
        // Try to borrow the value and panic on failure.
        // Panic is put into separate non-inlineable cold function
        // in order to not pollute this function
        // and hit compiler that this branch is unlikely.
        match self.try_borrow() {
            None => failed_to_borrow(),
            Some(r) => r,
        }
    }

    /// Mutably borrows the wrapped value, returning an error if the value is currently borrowed.
    ///
    /// The borrow lasts until the returned RefMut or all RefMuts derived from it exit scope.
    /// The value cannot be borrowed while this borrow is active.
    ///
    /// This is the non-panicking variant of borrow_mut.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::AtomicCell;
    ///
    /// let c = AtomicCell::new(5);
    ///
    /// {
    ///     let m = c.borrow();
    ///     assert!(c.try_borrow_mut().is_none());
    /// }
    ///
    /// assert!(c.try_borrow_mut().is_some());
    /// ```
    #[inline]
    pub fn try_borrow_mut(&self) -> Option<RefMut<'_, T>> {
        // Acquire shared borrow.
        match AtomicBorrowMut::try_new(&self.lock) {
            None => None,
            Some(borrow) => {
                // It is now safe to construct mutable borrow.
                Some(RefMut {
                    borrow,
                    value: unsafe {
                        // Locking mechanism ensures that mutable aliasing is impossible.
                        &mut *self.value.get()
                    },
                })
            }
        }
    }
    /// Mutably borrows the wrapped value.
    ///
    /// The borrow lasts until the returned [`RefMut`] or all [`RefMut`]s derived from it exit scope.
    /// The value cannot be borrowed while this borrow is active.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently borrowed. For a non-panicking variant, use try_borrow_mut.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::AtomicCell;
    ///
    /// let c = AtomicCell::new("hello".to_owned());
    ///
    /// *c.borrow_mut() = "bonjour".to_owned();
    ///
    /// assert_eq!(&*c.borrow(), "bonjour");
    /// ```
    ///
    /// An example of panic:
    ///
    /// ```should_panic
    /// use atomicell::AtomicCell;
    ///
    /// let c = AtomicCell::new(5);
    /// let m = c.borrow();
    ///
    /// let b = c.borrow_mut(); // this causes a panic
    /// ```
    #[inline(always)]
    #[track_caller]
    pub fn borrow_mut(&self) -> RefMut<'_, T> {
        // Try to borrow the value and panic on failure.
        // Panic is put into separate non-inlineable cold function
        // in order to not pollute this function
        // and hit compiler that this branch is unlikely.
        match self.try_borrow_mut() {
            None => failed_to_borrow_mut(),
            Some(r) => r,
        }
    }

    /// Returns a raw pointer to the underlying data in this cell.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::AtomicCell;
    ///
    /// let c = AtomicCell::new(5);
    ///
    /// let ptr = c.as_ptr();
    /// ```
    #[inline]
    pub const fn as_ptr(&self) -> *mut T {
        self.value.get()
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// This call borrows [`AtomicCell`] mutably (at compile-time) so there is no need for dynamic checks.
    ///
    /// However be cautious: this method expects self to be mutable,
    /// which is generally not the case when using a [`AtomicCell`]. Take a look at the borrow_mut method instead if self isn’t mutable.
    ///
    /// Also, please be aware that this method is only for special circumstances
    /// and is usually not what you want. In case of doubt, use borrow_mut instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::AtomicCell;
    ///
    /// let mut c = AtomicCell::new(5);
    /// *c.get_mut() += 1;
    ///
    /// assert_eq!(c, AtomicCell::new(6));
    /// ```
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.value.get_mut()
    }

    /// Undo the effect of leaked guards on the borrow state of the [`AtomicCell`].
    ///
    /// This call is similar to get_mut but more specialized.
    /// It borrows [`AtomicCell`] mutably to ensure no borrows exist and then resets the state tracking shared borrows.
    /// This is relevant if some Ref or RefMut borrows have been leaked.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::AtomicCell;
    ///
    /// let mut c = AtomicCell::new(0);
    /// core::mem::forget(c.borrow_mut());
    ///
    /// assert!(c.try_borrow().is_none());
    /// c.undo_leak();
    /// assert!(c.try_borrow().is_some());
    /// ```
    #[inline]
    pub fn undo_leak(&mut self) -> &mut T {
        *self.lock.get_mut() = 0;
        self.value.get_mut()
    }

    /// Immutably borrows the wrapped value, returning [`None`] if the value is currently mutably borrowed.
    ///
    /// # Safety
    ///
    /// Unlike [`AtomicCell::borrow`], this method is unsafe because it does not return a Ref,
    /// thus leaving the borrow flag untouched.
    ///
    /// Mutably borrowing the [`AtomicCell`] while the reference returned by this method is alive is undefined behaviour.
    ///
    /// [`AtomicCell::borrow`]: struct.AtomicCell.html#method.borrow
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::AtomicCell;
    ///
    /// let c = AtomicCell::new(5);
    ///
    /// {
    ///     assert!(unsafe { c.try_borrow_unguarded() }.is_some());
    ///     let m = c.borrow_mut();
    ///     assert!(unsafe { c.try_borrow_unguarded() }.is_none());
    /// }
    ///
    /// {
    ///     let m = c.borrow();
    ///     assert!(unsafe { c.try_borrow_unguarded() }.is_some());
    /// }
    /// ```
    #[inline]
    pub unsafe fn try_borrow_unguarded(&self) -> Option<&T> {
        if is_writing(self.lock.load(Ordering::Relaxed)) {
            None
        } else {
            // SAFETY: We check that nobody is actively writing now, but it is
            // the caller's responsibility to ensure that nobody writes until
            // the returned reference is no longer in use.
            Some(&*self.value.get())
        }
    }

    /// Mutably borrows the wrapped value, returning [`None`] if the value is currently mutably borrowed.
    ///
    /// # Safety
    ///
    /// Unlike [`AtomicCell::borrow_mut`], this method is unsafe because it does not return a Ref,
    /// thus leaving the borrow flag untouched.
    ///
    /// Borrowing the [`AtomicCell`] while the reference returned by this method is alive is undefined behaviour.
    ///
    /// [`AtomicCell::borrow_mut`]: struct.AtomicCell.html#method.borrow_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::AtomicCell;
    ///
    /// let c = AtomicCell::new(5);
    ///
    /// {
    ///     assert!(unsafe { c.try_borrow_unguarded_mut() }.is_some());
    ///     let m = c.borrow();
    ///     assert!(unsafe { c.try_borrow_unguarded_mut() }.is_none());
    /// }
    /// ```
    #[inline]
    pub unsafe fn try_borrow_unguarded_mut(&self) -> Option<&mut T> {
        let val = self.lock.load(Ordering::Relaxed);
        if is_reading(val) || is_writing(val) {
            None
        } else {
            // SAFETY: We check that nobody is actively reading or writing now, but it is
            // the caller's responsibility to ensure that nobody writes until
            // the returned reference is no longer in use.
            Some(&mut *self.value.get())
        }
    }
}

impl<T> AtomicCell<T>
where
    T: Default,
{
    /// Takes the wrapped value, leaving Default::default() in its place.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently borrowed.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::AtomicCell;
    ///
    /// let c = AtomicCell::new(5);
    /// let five = c.take();
    ///
    /// assert_eq!(five, 5);
    /// assert_eq!(c.into_inner(), 0);
    /// ```
    #[inline]
    pub fn take(&self) -> T {
        let mut r = self.borrow_mut();
        core::mem::take(&mut *r)
    }
}

impl<T> Clone for AtomicCell<T>
where
    T: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        let r = self.borrow();
        let t = Clone::clone(&*r);
        AtomicCell::new(t)
    }

    #[inline]
    fn clone_from(&mut self, other: &Self) {
        self.get_mut().clone_from(&other.borrow());
    }
}

impl<T> Debug for AtomicCell<T>
where
    T: Debug,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&*self.borrow(), f)
    }
}

impl<T> Display for AtomicCell<T>
where
    T: Display,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&*self.borrow(), f)
    }
}

impl<T> From<T> for AtomicCell<T> {
    #[inline]
    fn from(t: T) -> Self {
        AtomicCell::new(t)
    }
}

impl<T, U> PartialEq<AtomicCell<U>> for AtomicCell<T>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &AtomicCell<U>) -> bool {
        self.borrow().eq(&other.borrow())
    }
}

impl<T> Eq for AtomicCell<T> where T: Eq {}

impl<T, U> PartialOrd<AtomicCell<U>> for AtomicCell<T>
where
    T: PartialOrd<U>,
{
    fn partial_cmp(&self, other: &AtomicCell<U>) -> Option<cmp::Ordering> {
        self.borrow().partial_cmp(&other.borrow())
    }
}

impl<T> Ord for AtomicCell<T>
where
    T: Ord,
{
    fn cmp(&self, other: &AtomicCell<T>) -> cmp::Ordering {
        self.borrow().cmp(&other.borrow())
    }
}

#[inline(never)]
#[track_caller]
#[cold]
const fn too_many_refs() -> ! {
    panic!("Too many `Ref` instances created");
}

#[inline(never)]
#[track_caller]
#[cold]
const fn failed_to_borrow() -> ! {
    panic!("Failed to borrow AtomicCell immutably");
}

#[inline(never)]
#[track_caller]
#[cold]
const fn failed_to_borrow_mut() -> ! {
    panic!("Failed to borrow AtomicCell mutably");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_borrow() {
        let cell = AtomicCell::new(42);
        assert_eq!(*cell.borrow(), 42);
        *cell.borrow_mut() = 11;
        assert_eq!(*cell.borrow(), 11);
    }

    #[test]
    fn test_borrow_mut_split() {
        let cell = AtomicCell::new((1, 2));
        let (mut a, mut b) = RefMut::map_split(cell.borrow_mut(), |(a, b)| (a, b));
        *a += 5;
        *b += 5;

        assert!(cell.try_borrow().is_none());
        drop(a);
        assert!(cell.try_borrow().is_none());
        drop(b);

        assert_eq!(*cell.borrow(), (6, 7));
    }

    #[test]
    fn test_unsized() {
        let cell = AtomicCell::new([42; 3]);
        let cell: &AtomicCell<[i32]> = &cell;

        assert_eq!(cell.borrow().len(), 3);
        assert_eq!(cell.borrow()[0], 42);
        cell.borrow_mut()[0] = 11;
        assert_eq!(cell.borrow()[0], 11);
    }
}
