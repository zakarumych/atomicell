use core::{
    borrow::{Borrow, BorrowMut},
    cmp::Ordering,
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut, RangeBounds},
};

use crate::borrow::AtomicBorrowMut;

/// Wrapper for mutably borrowed [`AtomicCell`] that will released lock on drop.
///
/// This type can be dereferenced to [`&mut T`].
///
/// Implements [`Borrow<T>`], [`BorrowMut<T>`], [`AsRef<T>`] and [`AsMut<T>`] for convenience.
///
/// Implements [`Debug`], [`Display`], [`PartialEq`], [`PartialOrd`] and [`Hash`] by delegating to `T`.
///
/// [`AtomicCell`]: struct.AtomicCell.html
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
    fn partial_cmp(&self, other: &U) -> Option<Ordering> {
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
    /// [`AtomicCell`]: struct.AtomicCell.html
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

    /// Wraps external reference into [`RefMut`].
    /// And associated it with provided [`AtomicBorrowMut`]
    ///
    /// This function is intended to be used by [`AtomicCell`]
    /// or other abstractions that use `AtomicBorrow` for locking.
    ///
    /// [`AtomicCell`]: struct.AtomicCell.html
    ///
    /// # Examples
    ///
    /// ```
    /// use core::sync::atomic::AtomicIsize;
    /// use atomicell::{borrow::AtomicBorrowMut, RefMut};
    /// let counter = AtomicIsize::new(0);
    /// let borrow = AtomicBorrowMut::try_new(&counter).unwrap();
    ///
    /// let mut value = 42;
    /// let r = RefMut::with_borrow(&mut value, borrow);
    /// assert_eq!(*r, 42);
    /// ```
    #[inline]
    pub fn with_borrow(r: &'a mut T, borrow: AtomicBorrowMut<'a>) -> Self {
        RefMut { value: r, borrow }
    }

    /// Splits wrapper into two parts.
    /// One is reference to the value
    /// and the other is [`AtomicBorrowMut`] that guards it from being borrowed.
    ///
    /// # Safety
    ///
    /// User must ensure reference is not used after [`AtomicBorrowMut`] is dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, RefMut};
    ///
    /// let cell = AtomicCell::new(42);
    /// let r: RefMut<'_, i32> = cell.borrow_mut();
    ///
    /// unsafe {
    ///     let (r, borrow) = RefMut::into_split(r);
    ///     assert_eq!(*r, 42);
    ///
    ///     assert!(cell.try_borrow().is_none(), "Must not be able to borrow mutably yet");
    ///     assert!(cell.try_borrow_mut().is_none(), "Must not be able to borrow mutably yet");
    ///     drop(borrow);
    ///     assert!(cell.try_borrow_mut().is_some(), "Must be able to borrow mutably now");
    /// }
    /// ```
    #[inline]
    pub unsafe fn into_split(r: RefMut<'a, T>) -> (&'a mut T, AtomicBorrowMut<'a>) {
        (r.value, r.borrow)
    }

    /// Makes a new [`RefMut`] for a component of the borrowed data.
    ///
    /// The [`AtomicCell`] is already mutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as RefMut::map(...).
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// [`AtomicCell`]: struct.AtomicCell.html
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
    /// [`AtomicCell`]: struct.AtomicCell.html
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
    /// [`AtomicCell`]: struct.AtomicCell.html
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
    /// [`AtomicCell`]: struct.AtomicCell.html
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
    /// [`AtomicCell`]: struct.AtomicCell.html
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
    /// [`AtomicCell`]: struct.AtomicCell.html
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
    /// [`AtomicCell`]: struct.AtomicCell.html
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
    /// [`AtomicCell`]: struct.AtomicCell.html
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, RefMut};
    ///
    /// let c: &AtomicCell<[i32]> = &AtomicCell::new([1, 2, 3, 4, 5]);
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
