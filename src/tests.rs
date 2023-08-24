use super::{
    cell::AtomicCell,
    refs::{Ref, RefMut},
};

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

// For Miri to catch issues when calling a function.
//
// See how this scenerio affects std::cell::RefCell implementation:
// https://github.com/rust-lang/rust/issues/63787
//
// Also see relevant unsafe code guidelines issue:
// https://github.com/rust-lang/unsafe-code-guidelines/issues/125
#[test]
fn drop_and_borrow_in_fn_call() {
    {
        fn drop_and_borrow(cell: &AtomicCell<i32>, borrow: Ref<'_, i32>) {
            drop(borrow);
            *cell.borrow_mut() = 0;
        }

        let a = AtomicCell::new(0);
        let borrow = a.borrow();
        drop_and_borrow(&a, borrow);
    }

    {
        fn drop_and_borrow_mut(cell: &AtomicCell<i32>, borrow: RefMut<'_, i32>) {
            drop(borrow);
            *cell.borrow_mut() = 0;
        }

        let a = AtomicCell::new(0);
        let borrow = a.borrow_mut();
        drop_and_borrow_mut(&a, borrow);
    }
}
