use super::{cell::AtomicCell, refs::RefMut};

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
