use core::{
    borrow::{Borrow, BorrowMut},
    cmp::Ordering,
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Deref, DerefMut, RangeBounds},
    ptr::NonNull,
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
    value: NonNull<T>,
    borrow: AtomicBorrowMut<'a>,
    /// Makes [`RefMut`] invariant over T so that we can soundly allow mutation.
    _invariant: PhantomData<&'a mut T>,
}

// SAFETY: `Ref<'_, T> acts as a reference. `AtomicBorrowR` is a reference to an atomic.
unsafe impl<'b, T: ?Sized + 'b> Sync for RefMut<'b, T> where for<'a> &'a mut T: Sync {}
unsafe impl<'b, T: ?Sized + 'b> Send for RefMut<'b, T> where for<'a> &'a mut T: Send {}

impl<'a, T> Deref for RefMut<'a, T>
where
    T: ?Sized,
{
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        // SAFETY: We hold an exclusive lock on the pointer.
        unsafe { self.value.as_ref() }
    }
}

impl<'a, T> DerefMut for RefMut<'a, T>
where
    T: ?Sized,
{
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: We hold an exclusive lock on the pointer.
        unsafe { self.value.as_mut() }
    }
}

impl<'a, T> Debug for RefMut<'a, T>
where
    T: Debug,
{
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <T as Debug>::fmt(self, f)
    }
}

impl<'a, T> Display for RefMut<'a, T>
where
    T: Display + ?Sized,
{
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <T as Display>::fmt(self, f)
    }
}

impl<'a, T, U> PartialEq<U> for RefMut<'a, T>
where
    T: PartialEq<U> + ?Sized,
{
    #[inline(always)]
    fn eq(&self, other: &U) -> bool {
        <T as PartialEq<U>>::eq(self, other)
    }
}

impl<'a, T, U> PartialOrd<U> for RefMut<'a, T>
where
    T: PartialOrd<U> + ?Sized,
{
    #[inline(always)]
    fn partial_cmp(&self, other: &U) -> Option<Ordering> {
        <T as PartialOrd<U>>::partial_cmp(self, other)
    }
}

impl<'a, T> Hash for RefMut<'a, T>
where
    T: Hash + ?Sized,
{
    #[inline(always)]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        <T as Hash>::hash(self, state)
    }
}

impl<'a, T> Borrow<T> for RefMut<'a, T>
where
    T: ?Sized,
{
    #[inline(always)]
    fn borrow(&self) -> &T {
        self
    }
}

impl<'a, T> BorrowMut<T> for RefMut<'a, T>
where
    T: ?Sized,
{
    #[inline(always)]
    fn borrow_mut(&mut self) -> &mut T {
        self
    }
}

impl<'a, T, U> AsRef<U> for RefMut<'a, T>
where
    T: AsRef<U> + ?Sized,
{
    #[inline(always)]
    fn as_ref(&self) -> &U {
        <T as AsRef<U>>::as_ref(self)
    }
}

impl<'a, T, U> AsMut<U> for RefMut<'a, T>
where
    T: AsMut<U> + ?Sized,
{
    #[inline(always)]
    fn as_mut(&mut self) -> &mut U {
        <T as AsMut<U>>::as_mut(self)
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
            value: NonNull::from(r),
            borrow: AtomicBorrowMut::dummy(),
            _invariant: PhantomData,
        }
    }

    /// Wraps external reference into [`RefMut`].
    /// And associates it with provided [`AtomicBorrowMut`]
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
    /// use atomicell::{borrow::{AtomicBorrowMut, new_lock}, RefMut};
    /// let counter = new_lock();
    /// let borrow = AtomicBorrowMut::try_new(&counter).unwrap();
    ///
    /// let mut value = 42;
    /// let r = RefMut::with_borrow(&mut value, borrow);
    /// assert_eq!(*r, 42);
    /// ```
    #[inline]
    pub fn with_borrow(r: &'a mut T, borrow: AtomicBorrowMut<'a>) -> Self {
        RefMut {
            value: NonNull::from(r),
            borrow,
            _invariant: PhantomData,
        }
    }

    /// Splits wrapper into two parts. One is reference to the value and the other is
    /// [`AtomicBorrowMut`] that guards it from being borrowed.
    ///
    /// # Safety
    ///
    /// User must ensure [`NonNull`] is not dereferenced after [`AtomicBorrowMut`] is dropped.
    /// 
    /// You must also treat the [`NonNull`] as invariant over `T`. This means that any custom
    /// wrapper types you make around the [`NonNull<T>`] must also be invariant over `T`. This can
    /// be done by adding a [`PhantomData<*mut T>`] field to the struct.
    /// 
    /// See the source definition of [`RefMut`] for an example. 
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
    ///     assert_eq!(*r.as_ref(), 42);
    ///
    ///     assert!(cell.try_borrow().is_none(), "Must not be able to borrow mutably yet");
    ///     assert!(cell.try_borrow_mut().is_none(), "Must not be able to borrow mutably yet");
    ///     drop(borrow);
    ///     assert!(cell.try_borrow_mut().is_some(), "Must be able to borrow mutably now");
    /// }
    /// ```
    #[inline]
    pub fn into_split(r: RefMut<'a, T>) -> (NonNull<T>, AtomicBorrowMut<'a>) {
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
    pub fn map<F, U>(mut r: RefMut<'a, T>, f: F) -> RefMut<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
        U: ?Sized,
    {
        RefMut {
            value: NonNull::from(f(&mut *r)),
            borrow: r.borrow,
            _invariant: PhantomData,
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
        let RefMut { value, borrow, .. } = r;
        let _ptr = value.as_ptr();
        // SAFETY: function holds onto an exclusive reference for the duration
        // of its call through `r`, and the pointer is only de-referenced
        // inside of the function call never allowing the exclusive reference to
        // escape.
        match f(unsafe { &mut *_ptr }) {
            Some(value) => Ok(RefMut {
                value: NonNull::from(value),
                borrow,
                _invariant: PhantomData,
            }),
            None => {
                // SAFETY: same as above.
                Err(RefMut {
                    value,
                    borrow,
                    _invariant: PhantomData,
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
    pub fn map_split<U, V, F>(mut r: RefMut<'a, T>, f: F) -> (RefMut<'a, U>, RefMut<'a, V>)
    where
        U: ?Sized,
        V: ?Sized,
        F: FnOnce(&mut T) -> (&mut U, &mut V),
    {
        let borrow_u = r.borrow.clone();
        let borrow_v = r.borrow;

        // SAFETY: We hold an exclusive lock on the pointer.
        let (u, v) = f(unsafe { r.value.as_mut() });

        (
            RefMut {
                value: NonNull::from(u),
                borrow: borrow_u,
                _invariant: PhantomData,
            },
            RefMut {
                value: NonNull::from(v),
                borrow: borrow_v,
                _invariant: PhantomData,
            },
        )
    }

    /// Convert into a reference to the underlying data.
    ///
    /// The underlying [`AtomicCell`] can never be mutably borrowed from again
    /// and will always appear already mutably borrowed.
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
    pub fn leak(mut r: RefMut<'a, T>) -> &'a mut T {
        core::mem::forget(r.borrow);
        // SAFETY: We hold an exclusive lock on the pointer.
        unsafe { r.value.as_mut() }
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
    pub fn as_mut<U>(mut r: RefMut<'a, T>) -> RefMut<'a, U>
    where
        U: ?Sized,
        T: AsMut<U>,
    {
        RefMut {
            value: NonNull::from(<T as AsMut<U>>::as_mut(&mut *r)),
            borrow: r.borrow,
            _invariant: PhantomData,
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
    pub fn as_deref_mut(mut r: RefMut<'a, T>) -> RefMut<'a, T::Target>
    where
        T: DerefMut,
    {
        RefMut {
            value: NonNull::from(<T as DerefMut>::deref_mut(&mut *r)),
            borrow: r.borrow,
            _invariant: PhantomData,
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
    pub fn transpose(mut r: RefMut<'a, Option<T>>) -> Option<RefMut<'a, T>> {
        Some(RefMut {
            value: r.as_mut().map(NonNull::from)?,
            borrow: r.borrow,
            _invariant: PhantomData,
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
    pub fn slice<R>(mut r: RefMut<'a, [T]>, range: R) -> RefMut<'a, [T]>
    where
        R: RangeBounds<usize>,
    {
        let bounds = (range.start_bound().cloned(), range.end_bound().cloned());
        let slice = &mut *r;
        let slice = &mut slice[bounds];
        RefMut {
            value: NonNull::from(slice),
            borrow: r.borrow,
            _invariant: PhantomData,
        }
    }
}
