use crate::{ContentId, FileEntry, Hydration, Vfs, VfsError};

/// Pull-through source for a placeholder.
///
/// A full node, Unibus adapter, Kubo adapter, or iroh-blobs adapter implements
/// this. The VFS verifies the returned bytes against the registered content id.
pub trait Hydrator {
    fn fetch(&self, path: &str, content_id: &ContentId, size: u64) -> Result<Vec<u8>, VfsError>;
}

impl Vfs {
    /// Hydrate a placeholder from an external source and return the now-present
    /// entry. This is the cache pull-through primitive.
    pub fn hydrate_from<H: Hydrator>(
        &mut self,
        path: &str,
        hydrator: &H,
    ) -> Result<FileEntry, VfsError> {
        let entry = self
            .file(path)?
            .ok_or_else(|| VfsError::NotFound(path.into()))?;
        // A present flag is a claim about the namespace, not proof that the
        // content-addressed bytes survived. If the blob is missing, repair the
        // entry through the hydrator instead of returning an unreadable file.
        if entry.hydration == Hydration::Present && self.has_blob(&entry.content_id) {
            return Ok(entry);
        }
        let bytes = hydrator.fetch(&entry.path, &entry.content_id, entry.size)?;
        self.hydrate(&entry.path, &bytes)
    }

    /// Get a file, hydrating through the supplied source if this node only has
    /// a placeholder. Present files never call the source.
    pub fn get_or_hydrate<H: Hydrator>(
        &mut self,
        path: &str,
        hydrator: &H,
    ) -> Result<Vec<u8>, VfsError> {
        match self.get(path) {
            Ok(bytes) => Ok(bytes),
            Err(VfsError::NotHydrated(_)) => {
                self.hydrate_from(path, hydrator)?;
                self.get(path)
            }
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::{Expect, NodeRole};
    use std::cell::Cell;
    use tempfile::tempdir;

    struct Fixed(Vec<u8>, Cell<usize>);

    impl Hydrator for Fixed {
        fn fetch(
            &self,
            _path: &str,
            _content_id: &ContentId,
            _size: u64,
        ) -> Result<Vec<u8>, VfsError> {
            self.1.set(self.1.get() + 1);
            Ok(self.0.clone())
        }
    }

    #[test]
    fn a_present_entry_whose_bytes_are_missing_is_repaired_not_trusted() {
        let dir = tempdir().unwrap();
        let cas = dir.path().join("cas");
        let mut vfs = Vfs::open_with(dir.path(), NodeRole::Full, Some(&cas)).unwrap();
        let entry = vfs
            .put("notes.txt", b"hello", "text/plain", Expect::Any)
            .unwrap();
        // Simulate a GC or restore that removed the object but left the
        // namespace saying `present` — the exact stat/read mismatch.
        let blob = vfs.blob_path_for_test(&entry.content_id);
        std::fs::remove_file(&blob).unwrap();
        assert_eq!(
            vfs.file("notes.txt").unwrap().unwrap().hydration,
            Hydration::Present
        );
        assert!(matches!(
            vfs.get("notes.txt"),
            Err(VfsError::NotHydrated(_))
        ));

        let hydrator = Fixed(b"hello".to_vec(), Cell::new(0));
        assert_eq!(
            vfs.get_or_hydrate("notes.txt", &hydrator).unwrap(),
            b"hello"
        );
        assert_eq!(hydrator.1.get(), 1, "a missing present object must fetch");
        assert!(blob.is_file(), "the hydrated bytes land in the same CAS");
        assert_eq!(vfs.get("notes.txt").unwrap(), b"hello");
    }

    #[test]
    fn a_present_entry_with_bytes_does_not_call_the_hydrator() {
        let dir = tempdir().unwrap();
        let cas = dir.path().join("cas");
        let mut vfs = Vfs::open_with(dir.path(), NodeRole::Full, Some(&cas)).unwrap();
        vfs.put("notes.txt", b"hello", "text/plain", Expect::Any)
            .unwrap();
        let hydrator = Fixed(b"wrong".to_vec(), Cell::new(0));
        assert_eq!(
            vfs.get_or_hydrate("notes.txt", &hydrator).unwrap(),
            b"hello"
        );
        assert_eq!(hydrator.1.get(), 0);
    }
}
