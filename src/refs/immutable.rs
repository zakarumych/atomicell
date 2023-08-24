use core::{
    borrow::Borrow,
    cmp::Ordering,
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    ops::{Deref, RangeBounds},
    ptr::NonNull,
};

use crate::borrow::AtomicBorrow;

/// Wrapper for a borrowed [`AtomicCell`] that will released lock on drop.
///
/// This type can be dereferenced to [`&T`].
///
/// Implements [`Borrow<T>`] and [`AsRef<T>`] for convenience.
///
/// Implements [`Debug`], [`Display`], [`PartialEq`], [`PartialOrd`] and [`Hash`] by delegating to `T`.
///
/// [`AtomicCell`]: struct.AtomicCell.html
/// [`&T`]: https://doc.rust-lang.org/core/primitive.reference.html
pub struct Ref<'a, T: ?Sized> {
    value: NonNull<T>,
    borrow: AtomicBorrow<'a>,
}

// SAFETY: `Ref<'_, T> acts as a reference. `AtomicBorrowR` is a reference to an atomic.
unsafe impl<'b, T: ?Sized + 'b> Sync for Ref<'b, T> where for<'a> &'a T: Sync {}
unsafe impl<'b, T: ?Sized + 'b> Send for Ref<'b, T> where for<'a> &'a T: Send {}

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
        // SAFETY: We hold a shared borrow lock.
        unsafe { self.value.as_ref() }
    }
}

impl<'a, T> Debug for Ref<'a, T>
where
    T: Debug + ?Sized,
{
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <T as Debug>::fmt(self, f)
    }
}

impl<'a, T> Display for Ref<'a, T>
where
    T: Display + ?Sized,
{
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <T as Display>::fmt(self, f)
    }
}

impl<'a, T, U> PartialEq<U> for Ref<'a, T>
where
    T: PartialEq<U> + ?Sized,
{
    #[inline(always)]
    fn eq(&self, other: &U) -> bool {
        <T as PartialEq<U>>::eq(self, other)
    }
}

impl<'a, T, U> PartialOrd<U> for Ref<'a, T>
where
    T: PartialOrd<U> + ?Sized,
{
    #[inline(always)]
    fn partial_cmp(&self, other: &U) -> Option<Ordering> {
        <T as PartialOrd<U>>::partial_cmp(self, other)
    }
}

impl<'a, T> Hash for Ref<'a, T>
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

impl<'a, T> Borrow<T> for Ref<'a, T> {
    #[inline(always)]
    fn borrow(&self) -> &T {
        self
    }
}

impl<'a, T, U> AsRef<U> for Ref<'a, T>
where
    T: AsRef<U> + ?Sized,
{
    #[inline(always)]
    fn as_ref(&self) -> &U {
        <T as AsRef<U>>::as_ref(self)
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
    /// [`AtomicCell`]: struct.AtomicCell.html
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
            value: NonNull::from(r),
            borrow: AtomicBorrow::dummy(),
        }
    }

    /// Wraps external reference into [`Ref`].
    /// And associates it with provided [`AtomicBorrow`]
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
    /// use atomicell::{borrow::{AtomicBorrow, new_lock}, Ref};
    /// let counter = new_lock();
    /// let borrow = AtomicBorrow::try_new(&counter).unwrap();
    ///
    /// let r = Ref::with_borrow(&42, borrow);
    /// assert_eq!(*r, 42);
    /// ```
    #[inline]
    pub fn with_borrow(r: &'a T, borrow: AtomicBorrow<'a>) -> Self {
        Ref {
            value: NonNull::from(r),
            borrow,
        }
    }

    /// Splits wrapper into two parts. One is reference to the value and the other is
    /// [`AtomicBorrow`] that guards it from being borrowed mutably.
    ///
    /// # Safety
    ///
    /// User must ensure [`NonNull`] is not dereferenced after [`AtomicBorrow`] is dropped.
    ///
    /// Also, the [`NonNull<T>`] that is returned is still only valid for reads, not writes.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, Ref};
    ///
    /// let cell = AtomicCell::new(42);
    /// let r: Ref<'_, i32> = cell.borrow();
    ///
    /// unsafe {
    ///     let (r, borrow) = Ref::into_split(r);
    ///     assert_eq!(*r.as_ref(), 42);
    ///
    ///     assert!(cell.try_borrow().is_some(), "Must be able to borrow immutably");
    ///     assert!(cell.try_borrow_mut().is_none(), "Must not be able to borrow mutably yet");
    ///     drop(borrow);
    ///     assert!(cell.try_borrow_mut().is_some(), "Must be able to borrow mutably now");
    /// }
    /// ```
    #[inline]
    pub fn into_split(r: Ref<'a, T>) -> (NonNull<T>, AtomicBorrow<'a>) {
        (r.value, r.borrow)
    }

    /// Makes a new [`Ref`] for a component of the borrowed data.
    ///
    /// The [`AtomicCell`] is already immutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as Ref::map(...).
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// [`AtomicCell`]: struct.AtomicCell.html
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
            value: NonNull::from(f(&*r)),
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
    /// [`AtomicCell`]: struct.AtomicCell.html
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
        match f(&*r) {
            Some(value) => Ok(Ref {
                value: NonNull::from(value),
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
    /// [`AtomicCell`]: struct.AtomicCell.html
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

        // SAFETY: we have a shared reference lock on the pointer.
        let (u, v) = f(unsafe { r.value.as_ref() });

        (
            Ref {
                value: NonNull::from(u),
                borrow: borrow_u,
            },
            Ref {
                value: NonNull::from(v),
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
    /// [`AtomicCell`]: struct.AtomicCell.html
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
        // SAFETY: we have a shared reference lock on the pointer.
        unsafe { r.value.as_ref() }
    }

    /// Converts reference and returns result wrapped in the [`Ref`].
    ///
    /// The [`AtomicCell`] is already immutably borrowed, so this cannot fail.
    ///
    /// This is an associated function that needs to be used as `Ref::map_split(...)`.
    /// A method would interfere with methods of the same name on the contents of a [`AtomicCell`] used through [`Deref`].
    ///
    /// [`AtomicCell`]: struct.AtomicCell.html
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
            value: NonNull::from(<T as AsRef<U>>::as_ref(&r)),
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
    /// [`AtomicCell`]: struct.AtomicCell.html
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
            value: NonNull::from(<T as Deref>::deref(&*r)),

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
    /// [`AtomicCell`]: struct.AtomicCell.html
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
            value: r.as_ref().map(NonNull::from)?,
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
    /// [`AtomicCell`]: struct.AtomicCell.html
    ///
    /// # Examples
    ///
    /// ```
    /// use atomicell::{AtomicCell, Ref};
    ///
    /// let c: &AtomicCell<[i32]> = &AtomicCell::new([1, 2, 3, 4, 5]);
    /// let b1: Ref<[i32]> = c.borrow();
    /// let b2: Ref<[i32]> = Ref::slice(b1, 2..4);
    /// assert_eq!(*b2, [3, 4])
    /// ```
    #[inline]
    pub fn slice<R>(r: Ref<'a, [T]>, range: R) -> Ref<'a, [T]>
    where
        R: RangeBounds<usize>,
    {
        let bounds = (range.start_bound().cloned(), range.end_bound().cloned());
        let slice = &*r;
        let slice = &slice[bounds];
        Ref {
            value: NonNull::from(slice),
            borrow: r.borrow,
        }
    }
}
