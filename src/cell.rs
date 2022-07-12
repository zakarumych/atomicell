use core::{
    cell::UnsafeCell,
    cmp,
    fmt::{self, Debug, Display},
    sync::atomic::{AtomicIsize, Ordering},
};

use crate::{
    borrow::{is_reading, is_writing, AtomicBorrow, AtomicBorrowMut},
    refs::{Ref, RefMut},
};

/// A mutable memory location with dynamically checked borrow rules
/// This type behaves mostly like [`core::cell::RefCell`].
/// The main difference is that this type uses atomic operations for borrowing.
/// Thus allowing to use it in multi-threaded environment.
///
/// [`AtomicCell<T>`] implements [`Send`] if `T: Send`.
/// [`AtomicCell<T>`] implements [`Sync`] if `T: Send + Sync`.
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
        // TODO: Add `const` when `UnsafeCell::into_inner` is stabilized as const.
        self.value.into_inner()
    }

    /// Replaces the wrapped value with a new one, returning the old value, without deinitializing either one.
    ///
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

    /// Replaces the wrapped value with a new one computed from f,
    /// returning the old value,
    /// without deinitializing either one.
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
                let t = f(&mut *borrow);
                core::mem::replace(&mut *borrow, t)
            }
        }
    }

    /// Swaps the wrapped value of self with the wrapped value of other,
    /// without deinitializing either one.
    ///
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
            Some(mut borrow) => match other.try_borrow_mut() {
                None => failed_to_borrow_mut(),
                Some(mut other_borrow) => {
                    core::mem::swap(&mut *borrow, &mut *other_borrow);
                }
            },
        }
    }
}

impl<T> AtomicCell<T>
where
    T: ?Sized,
{
    /// Immutably borrows the wrapped value,
    /// returning [`None`] if the value is currently mutably borrowed.
    ///
    /// The borrow lasts until the returned [`Ref`], all [`Ref`]s derived from it and all its clones exit scope.
    ///
    /// Multiple immutable borrows can be taken out at the same time.
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
                let r = unsafe {
                    // Locking mechanism ensures that mutable aliasing is impossible.
                    &*self.value.get()
                };

                Some(Ref::with_borrow(r, borrow))
            }
        }
    }

    /// Immutably borrows the wrapped value.
    ///
    /// The borrow lasts until the returned [`Ref`], all [`Ref`]s derived from it and all its clones exit scope.
    ///
    /// Multiple immutable borrows can be taken out at the same time.
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
    /// The borrow lasts until the returned [`RefMut`] or all [`RefMut`]s derived from it exit scope.
    ///
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
                let r = unsafe {
                    // Locking mechanism ensures that mutable aliasing is impossible.
                    &mut *self.value.get()
                };

                Some(RefMut::with_borrow(r, borrow))
            }
        }
    }
    /// Mutably borrows the wrapped value.
    ///
    /// The borrow lasts until the returned [`RefMut`] or all [`RefMut`]s derived from it exit scope.
    ///
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
    /// which is generally not the case when using a [`AtomicCell`].
    /// Take a look at the [borrow_mut] method instead if self isn’t mutable.
    ///
    /// Also, please be aware that this method is only for special circumstances
    /// and is usually not what you want. In case of doubt, use borrow_mut instead.
    ///
    /// [borrow_mut]: #method.borrow_mut
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
    /// Unlike [borrow], this method is unsafe because it does not return a Ref,
    /// thus leaving the borrow flag untouched.
    ///
    /// Mutably borrowing the [`AtomicCell`] while the reference returned by this method is alive is undefined behaviour.
    ///
    /// [borrow]: #method.borrow
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
    /// Unlike [borrow_mut], this method is unsafe because it does not return a Ref,
    /// thus leaving the borrow flag untouched.
    ///
    /// Borrowing the [`AtomicCell`] while the reference returned by this method is alive is undefined behaviour.
    ///
    /// [borrow_mut]: #method.borrow_mut
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
    /// Takes the wrapped value, leaving [`Default::default()`] in its place.
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
const fn failed_to_borrow() -> ! {
    panic!("Failed to borrow AtomicCell immutably");
}

#[inline(never)]
#[track_caller]
#[cold]
const fn failed_to_borrow_mut() -> ! {
    panic!("Failed to borrow AtomicCell mutably");
}
