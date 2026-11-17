//! Character-level merging inside one block.
//!
//! [`crate::crdt`] merges whole blocks and says out loud that it cannot merge
//! inside one: two edits to the same paragraph resolve last-writer-wins, so one
//! of them is lost. This module is what replaces that rule. A block's text becomes
//! a sequence of characters that each carry an identity, and an insert names the
//! character it follows — so two replicas that edited different parts of the same
//! paragraph both keep their edit, whichever order the ops arrive in.
//!
//! The scheme is RGA (Replicated Growable Array), chosen because it needs nothing
//! this codebase does not already have: an identity per character in the same
//! `counter@actor` space as everything else, and a tree of "after" links whose
//! traversal is the order. There is no central sequencer and no timestamp
//! comparison — position is identity, exactly as it is for blocks.
//!
//! Three limits, stated rather than discovered:
//!
//! - A character is a `char`, not a grapheme cluster. Two people editing the two
//!   halves of an emoji will produce one emoji, which is what a byte-level merge
//!   does to everything and what a grapheme-aware CRDT would fix later.
//! - Tombstones are kept: deleting a character marks it, so an insert anchored to
//!   it still finds its place. The cost is that a long-lived document carries the
//!   history of its deletions, which is a compaction problem, not a correctness one.
//! - Anchors are never garbage collected, so an op naming a character this replica
//!   has not seen is deferred, not dropped — the same rule blocks follow.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::crdt::BlockId;

/// One character, at one position, for ever.
pub type CharId = BlockId;

/// What a replica learned about one block's text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "text_op", rename_all = "snake_case")]
pub enum TextOp {
    /// A character placed after another. `after: None` puts it at the head.
    Insert {
        id: CharId,
        after: Option<CharId>,
        value: char,
    },
    /// A character removed. It keeps its place as a tombstone.
    Delete { id: CharId },
}

/// What applying a text op did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextApplied {
    Duplicate,
    Applied,
    /// The op anchors to a character this replica has not seen yet, or names one
    /// it does not hold.
    Deferred,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Node {
    after: Option<CharId>,
    value: char,
    #[serde(default)]
    deleted: bool,
}

/// A block's text, as one replica sees it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Text {
    #[serde(default)]
    nodes: BTreeMap<CharId, Node>,
}

impl Text {
    /// Seed a block's text from the text a replica already has.
    ///
    /// Seeding has to be deterministic or it defeats the point: two machines that
    /// start from the same block text must mint the same identities, or every
    /// character looks new on both and a merge doubles the paragraph. The block is
    /// the same block on both sides, so its characters are named from the block's
    /// own identity and their position in it.
    ///
    /// The seeded actor is *not* the block's actor: seeded characters are older than
    /// anything a person types, so they get an actor nothing else writes under. A
    /// live op can then never collide with a seed, and an insert at the head — the
    /// newest child of an anchor — sorts in front of the seeded first character,
    /// which is what "insert at position zero" has to mean.
    #[must_use]
    pub fn seeded(block: &BlockId, text: &str) -> Self {
        let actor = format!("{}#{}", block.actor, block.counter);
        let mut value = Self::default();
        let mut previous: Option<CharId> = None;
        for (index, ch) in text.chars().enumerate() {
            let id = CharId::new(index as u64, &actor);
            value.nodes.insert(
                id.clone(),
                Node {
                    after: previous.clone(),
                    value: ch,
                    deleted: false,
                },
            );
            previous = Some(id);
        }
        value
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The characters in order, tombstones skipped.
    #[must_use]
    pub fn render(&self) -> String {
        self.order()
            .into_iter()
            .filter(|(_, node)| !node.deleted)
            .map(|(_, node)| node.value)
            .collect()
    }

    /// As [`render`](Self::render), but keeping the deleted characters so a caller
    /// can see what a tombstone holds.
    #[must_use]
    pub fn chars(&self) -> Vec<char> {
        self.order()
            .into_iter()
            .filter(|(_, node)| !node.deleted)
            .map(|(_, node)| node.value)
            .collect()
    }

    /// Every live character's id, in order.
    #[must_use]
    pub fn live_ids(&self) -> Vec<CharId> {
        self.order()
            .into_iter()
            .filter(|(_, node)| !node.deleted)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Document order: a depth-first walk of the `after` tree, siblings newest
    /// first, so an insert lands immediately after the character it names.
    #[must_use]
    fn order(&self) -> Vec<(&CharId, &Node)> {
        let mut children: BTreeMap<Option<&CharId>, BTreeSet<&CharId>> = BTreeMap::new();
        for (id, node) in &self.nodes {
            let anchor = node
                .after
                .as_ref()
                .filter(|after| self.nodes.contains_key(*after));
            children.entry(anchor).or_default().insert(id);
        }
        let mut out = Vec::with_capacity(self.nodes.len());
        let mut seen = BTreeSet::new();
        let mut stack: Vec<Vec<&CharId>> = Vec::new();
        if let Some(roots) = children.get(&None) {
            stack.push(roots.iter().copied().collect());
        }
        while let Some(frame) = stack.last_mut() {
            let Some(id) = frame.pop() else {
                stack.pop();
                continue;
            };
            if !seen.insert(id) {
                continue;
            }
            out.push((id, &self.nodes[id]));
            if let Some(next) = children.get(&Some(id)) {
                stack.push(next.iter().copied().collect());
            }
        }
        // A cycle cannot come from honest ops, but a corrupt record must not make
        // characters vanish.
        for (id, node) in &self.nodes {
            if !seen.contains(id) {
                out.push((id, node));
            }
        }
        out
    }

    pub fn apply(&mut self, op: &TextOp) -> TextApplied {
        match op {
            TextOp::Insert { id, after, value } => {
                if self.nodes.contains_key(id) {
                    return TextApplied::Duplicate;
                }
                if let Some(after) = after {
                    if !self.nodes.contains_key(after) {
                        return TextApplied::Deferred;
                    }
                }
                self.nodes.insert(
                    id.clone(),
                    Node {
                        after: after.clone(),
                        value: *value,
                        deleted: false,
                    },
                );
                TextApplied::Applied
            }
            TextOp::Delete { id } => match self.nodes.get_mut(id) {
                None => TextApplied::Deferred,
                Some(node) if node.deleted => TextApplied::Duplicate,
                Some(node) => {
                    node.deleted = true;
                    TextApplied::Applied
                }
            },
        }
    }

    /// Apply a batch in any order, retrying what arrived early.
    pub fn apply_all(&mut self, ops: &[TextOp]) -> Vec<TextOp> {
        let mut pending: Vec<&TextOp> = ops.iter().collect();
        loop {
            let count = pending.len();
            let mut deferred = Vec::new();
            for op in pending {
                if self.apply(op) == TextApplied::Deferred {
                    deferred.push(op);
                }
            }
            if deferred.is_empty() || deferred.len() == count {
                return deferred.into_iter().cloned().collect();
            }
            pending = deferred;
        }
    }

    /// Replace the characters in `[start, start + remove)` with `insert`.
    ///
    /// This is the seam a file save goes through: a person editing markdown changes
    /// a few characters, and the ops say which. `next` mints identities for the new
    /// characters, and the caller keeps it ahead of everything it has seen.
    pub fn splice(
        &mut self,
        start: usize,
        remove: usize,
        insert: &str,
        next: &mut dyn FnMut() -> CharId,
    ) -> Vec<TextOp> {
        let live = self.live_ids();
        let mut ops = Vec::new();
        // The anchor is the character before the changed run. A change at the head
        // anchors to nothing, which is the head of the text.
        let mut after = if start == 0 {
            None
        } else {
            live.get(start - 1).cloned()
        };
        for id in live.iter().skip(start).take(remove) {
            ops.push(TextOp::Delete { id: id.clone() });
        }
        for ch in insert.chars() {
            let id = next();
            ops.push(TextOp::Insert {
                id: id.clone(),
                after: after.clone(),
                value: ch,
            });
            after = Some(id);
        }
        ops
    }

    /// How different two renderings are: the changed run as `(start, removed,
    /// inserted)`, computed from the common prefix and suffix.
    ///
    /// A character-level diff is the honest shape here and the cheap one: the ops a
    /// save produces are proportional to what actually changed, not to the size of
    /// the paragraph. A moved sentence therefore reads as a delete plus an insert,
    /// which is what it is.
    #[must_use]
    pub fn difference(old: &str, new: &str) -> (usize, usize, String) {
        let old_chars: Vec<char> = old.chars().collect();
        let new_chars: Vec<char> = new.chars().collect();
        let mut prefix = 0;
        while prefix < old_chars.len()
            && prefix < new_chars.len()
            && old_chars[prefix] == new_chars[prefix]
        {
            prefix += 1;
        }
        let mut suffix = 0;
        while suffix < old_chars.len().saturating_sub(prefix)
            && suffix < new_chars.len().saturating_sub(prefix)
            && old_chars[old_chars.len() - 1 - suffix] == new_chars[new_chars.len() - 1 - suffix]
        {
            suffix += 1;
        }
        let removed = old_chars.len() - prefix - suffix;
        let inserted: String = new_chars[prefix..new_chars.len() - suffix].iter().collect();
        (prefix, removed, inserted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block() -> BlockId {
        BlockId::new(7, "neo")
    }

    /// A block's text, plus the next identity a local writer will use.
    fn seeded(text: &str) -> (Text, u64) {
        (Text::seeded(&block(), text), 100)
    }

    #[test]
    fn seeding_is_deterministic_across_replicas() {
        // Two machines given the same block and the same text mint the same ids, or
        // every character would look new on both sides of a merge.
        let (left, _) = seeded("hello");
        let (right, _) = seeded("hello");
        assert_eq!(left.live_ids(), right.live_ids());
        assert_eq!(left.render(), "hello");
    }

    #[test]
    fn two_edits_inside_one_paragraph_both_survive() {
        // The claim this module exists for. Two machines edit different words of one
        // paragraph; a whole-block rule keeps one of them.
        let base = "the quick brown fox";
        let (mut neo, mut clock) = seeded(base);
        let (mut studio, mut other) = seeded(base);

        let start = base.find("quick").unwrap();
        let neo_ops = neo.splice(start, 0, "very ", &mut || {
            clock += 1;
            CharId::new(clock, "neo")
        });
        neo.apply_all(&neo_ops);

        let busy = base.find("brown").unwrap();
        let studio_ops = studio.splice(busy, 5, "red", &mut || {
            other += 1;
            CharId::new(other, "studio")
        });
        studio.apply_all(&studio_ops);

        // Each learns the other's ops.
        let deferred = neo.apply_all(&studio_ops);
        assert!(
            deferred.is_empty(),
            "nothing should be waiting: {deferred:?}"
        );
        studio.apply_all(&neo_ops);

        assert_eq!(neo.render(), studio.render());
        assert_eq!(neo.render(), "the very quick red fox");
    }

    #[test]
    fn a_deleted_run_stays_anchored_for_later_inserts() {
        let base = "abcdef";
        let (mut text, mut clock) = seeded(base);
        let mut next = move || {
            clock += 1;
            CharId::new(clock, "neo")
        };
        // Remove "cd", then put a character where they were.
        let remove = text.splice(2, 2, "", &mut next);
        text.apply_all(&remove);
        assert_eq!(text.render(), "abef");
        let insert = text.splice(2, 0, "X", &mut next);
        text.apply_all(&insert);
        assert_eq!(text.render(), "abXef");
    }

    #[test]
    fn an_op_that_arrives_before_its_anchor_is_deferred_not_dropped() {
        let (mut text, _) = seeded("ab");
        let head = CharId::new(1, "studio");
        let tail = CharId::new(2, "studio");
        let anchored = TextOp::Insert {
            id: tail.clone(),
            after: Some(head.clone()),
            value: '!',
        };
        let at_head = TextOp::Insert {
            id: head,
            after: None,
            value: '>',
        };
        // Newest-first arrival is normal, and the anchored op names a character
        // that is not here yet.
        assert_eq!(text.apply(&anchored), TextApplied::Deferred);
        assert!(text.apply_all(&[anchored, at_head]).is_empty());
        assert_eq!(text.render(), ">!ab");
    }

    #[test]
    fn a_save_sends_only_what_changed() {
        let old = "one two three";
        let new = "one 2 three";
        let (start, removed, inserted) = Text::difference(old, new);
        assert_eq!((start, removed, inserted.as_str()), (4, 3, "2"));
        let (mut text, mut clock) = seeded(old);
        let ops = text.splice(start, removed, &inserted, &mut || {
            clock += 1;
            CharId::new(clock, "neo")
        });
        text.apply_all(&ops);
        assert_eq!(text.render(), new);
        assert_eq!(ops.len(), 4, "one delete per character plus the insert");
    }

    #[test]
    fn the_same_op_twice_is_a_duplicate() {
        let (mut text, _) = seeded("ab");
        let op = TextOp::Insert {
            id: CharId::new(5, "neo"),
            after: None,
            value: 'X',
        };
        assert_eq!(text.apply(&op), TextApplied::Applied);
        assert_eq!(text.apply(&op), TextApplied::Duplicate);
        assert_eq!(text.render(), "Xab");
    }
}
