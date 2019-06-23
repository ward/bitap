//! Library for approximate string matching (fuzzy searching) using the bitap algorithm.
//! To search, first create a Bitap struct. This enables you to set options. Builder pattern of
//! sorts. Once you have that done, you can use bitap() or search() to actually search.
//!
//! ```
//! let bitap = Bitap::new().distance(200).threshold(0.4);
//! bitap.bitap("hello world", "elo");
//! ```

use std::collections::HashMap;

pub struct Bitap {
    distance: i64,
    expected_location: i64,
    find_all_matches: bool,
    min_match_char_length: i64,
    threshold: f64,
}
impl Bitap {
    /// Create Bitap configuration struct
    pub fn new() -> Bitap {
        std::default::Default::default()
    }

    /// Set the penalty a match receives when it is further away from the expected location in the
    /// text. A higher number means a lower penalty for being further away.
    pub fn distance(self, distance: i64) -> Bitap {
        Bitap { distance, ..self }
    }
    pub fn expected_location(self, expected_location: i64) -> Bitap {
        Bitap {
            expected_location,
            ..self
        }
    }
    pub fn find_all_matches(self, find_all_matches: bool) -> Bitap {
        Bitap {
            find_all_matches,
            ..self
        }
    }
    pub fn min_match_char_length(self, min_match_char_length: i64) -> Bitap {
        Bitap {
            min_match_char_length,
            ..self
        }
    }
    /// The threshold (in the bitap score, part of the result) to consider something a match. The
    /// lower this number, the more something needs to match. Ranges from 0 to 1.
    pub fn threshold(self, threshold: f64) -> Bitap {
        Bitap { threshold, ..self }
    }

    /// Looks for pattern in several texts. Returns a vector of texts for which a match was found.
    /// Texts with the best matches come first.
    pub fn search<'a, I>(&self, pattern: &str, haystack: I) -> Vec<&'a str>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let mut matches = vec![];
        for s in haystack {
            let result = self.bitap(s, pattern);
            if result.is_match {
                matches.push((result.score, s));
            }
        }
        matches.sort_by(|(score, _s), (score2, _s2)| score.partial_cmp(score2).unwrap());
        matches.iter().map(|(_score, s)| *s).collect()
    }

    /// Calculate approximate string match (fuzzy search) using the bitap search algorithm.
    /// Implementation based on the implementation by the Fuse JavaScript library at
    /// https://github.com/krisk/Fuse/blob/master/src/bitap/bitap_search.js.
    pub fn bitap(&self, text: &str, pattern: &str) -> SearchResult {
        let textlength = text.len() as i64;
        let patternlength = pattern.len() as i64;

        // If no pattern provided, just say no match
        // Could also make a case for full match?
        if patternlength == 0 {
            return SearchResult {
                is_match: false,
                score: 1.0,
                matched_indices: vec![],
            };
        }

        let pattern_alphabet = Bitap::pattern_alphabet(pattern);

        // mask of the matches
        let mut match_mask = vec![0; textlength as usize];

        let expected_location = self.expected_location;
        let distance = self.distance;
        let mut currentthreshold = self.threshold;

        // Speed up by checking for an exact match
        if let Some(found_idx) = text.find(pattern) {
            let score =
                Bitap::bitapscore(pattern, 0, found_idx as i64, expected_location, distance);
            if score < currentthreshold {
                currentthreshold = score;
            }

            if let Some(rfound_idx) = text.rfind(pattern) {
                let score =
                    Bitap::bitapscore(pattern, 0, rfound_idx as i64, expected_location, distance);
                if score < currentthreshold {
                    currentthreshold = score;
                }
            }
        }
        let mut best_location = -1;

        let mut last_bits: Vec<i64> = vec![0; text.len() + pattern.len() + 2]; // TODO just trying to make this work, is this ok?
        let mut finalscore = 1.0;
        let mut binmax = textlength + patternlength;

        let mask = 1 << (patternlength - 1);

        for i in 0..patternlength {
            let mut binmin = 0;
            let mut binmid = binmax;

            while binmin < binmid {
                let score = Bitap::bitapscore(
                    pattern,
                    i,
                    expected_location + binmid,
                    expected_location,
                    distance,
                );
                if score <= currentthreshold {
                    binmin = binmid;
                } else {
                    binmax = binmid;
                }
                binmid = (binmax - binmin) / 2 + binmin;
            }
            // result of the while becomes maximum for next iteration
            binmax = binmid;

            let mut start = std::cmp::max(1, expected_location - binmid + 1);
            let finish = if self.find_all_matches {
                textlength
            } else {
                std::cmp::min(expected_location + binmid, textlength) + patternlength
            };

            // init bit array
            let mut bitarr = vec![0; (finish + 2) as usize];
            bitarr[finish as usize + 1] = (1 << i) - 1;

            let mut j = finish as usize;
            while j >= start as usize {
                let currentlocation = j - 1;
                let charidx = &text.chars().nth(currentlocation as usize);
                let charmatch = *if charidx.is_some() {
                    pattern_alphabet.get(&charidx.unwrap()).unwrap_or(&0)
                } else {
                    &0
                };
                if charmatch > 0 {
                    match_mask[currentlocation as usize] = 1;
                }

                // first pass exact match
                bitarr[j] = ((bitarr[j + 1] << 1) | 1) & charmatch;

                // subsequent passes fuzzy match
                if i != 0 {
                    bitarr[j] |= (((last_bits[j + 1] | last_bits[j]) << 1) | 1) | last_bits[j + 1];
                }

                if bitarr[j] & mask > 0 {
                    finalscore = Bitap::bitapscore(
                        pattern,
                        i,
                        currentlocation as i64,
                        expected_location,
                        distance,
                    );

                    if finalscore <= currentthreshold {
                        currentthreshold = finalscore;
                        best_location = currentlocation as i64;

                        if best_location <= expected_location {
                            break;
                        }

                        start = std::cmp::max(1, 2 * expected_location - best_location as i64);
                    }
                }

                j -= 1;
            } // end while

            let score = Bitap::bitapscore(
                pattern,
                i + 1,
                expected_location,
                expected_location,
                distance,
            );

            if score > currentthreshold {
                break;
            }

            last_bits = bitarr;
        }

        SearchResult {
            is_match: best_location >= 0,
            score: if finalscore == 0.0 { 0.001 } else { finalscore },
            matched_indices: Bitap::matched_indices(match_mask, self.min_match_char_length),
        }
    }

    fn bitapscore(
        pattern: &str,
        errors: i64,
        currentlocation: i64,
        expected_location: i64,
        distance: i64,
    ) -> f64 {
        let accuracy = errors as f64 / pattern.len() as f64;
        let proximity = (expected_location as f64 - currentlocation as f64).abs();

        if distance == 0 {
            if proximity as i64 != 0 {
                1.0
            } else {
                accuracy
            }
        } else {
            accuracy + (proximity / distance as f64)
        }
    }

    /// Creates mapping of the characters in a pattern to numbers
    fn pattern_alphabet(pattern: &str) -> HashMap<char, i64> {
        let mut mask = HashMap::new();
        let patternlength = pattern.len();
        for c in pattern.chars() {
            mask.insert(c, 0);
        }
        for (i, c) in pattern.chars().enumerate() {
            mask.entry(c)
                .and_modify(|charmask| *charmask |= 1 << (patternlength - i - 1));
        }
        mask
    }

    fn matched_indices(matchmask: Vec<i64>, min_match_char_length: i64) -> Vec<(i64, i64)> {
        let mut matched_indices = vec![];
        let mut start = -1;
        let mut i = 0;
        let matchmask_len = matchmask.len() as i64;

        while i < matchmask_len {
            let a_match = matchmask[i as usize];
            if a_match != 0 && start == -1 {
                start = i
            } else if a_match == 0 && start != -1 {
                let end = i - 1;
                if (end - start) + 1 >= min_match_char_length {
                    matched_indices.push((start, end));
                }
                start = -1;
            }

            i += 1;
        }

        if matchmask[i as usize - 1] != 0 && (i - start) >= min_match_char_length {
            matched_indices.push((start, i - 1));
        }

        matched_indices
    }
}

impl std::default::Default for Bitap {
    fn default() -> Self {
        Bitap {
            expected_location: 0,
            distance: 100,
            threshold: 0.6,
            find_all_matches: false,
            min_match_char_length: 1,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct SearchResult {
    pub is_match: bool,
    pub score: f64,
    pub matched_indices: Vec<(i64, i64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn equal_hashmaps<K, V>(h1: &HashMap<K, V>, h2: &HashMap<K, V>) -> bool
    where
        K: std::hash::Hash + std::cmp::Eq,
        V: std::cmp::PartialEq,
    {
        for key in h1.keys().chain(h2.keys()) {
            let v1 = h1.get(key);
            let v2 = h2.get(key);
            if v1.is_none() || v2.is_none() || v1.unwrap() != v2.unwrap() {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_pattern_alphabet() {
        let mut hm = HashMap::new();
        hm.insert('h', 16);
        hm.insert('e', 8);
        hm.insert('l', 6);
        hm.insert('o', 1);
        assert!(equal_hashmaps(&Bitap::pattern_alphabet("hello"), &hm));
        let mut hm = HashMap::new();
        hm.insert('w', 4096);
        hm.insert('a', 2056);
        hm.insert('r', 1026);
        hm.insert('d', 512);
        hm.insert(' ', 256);
        hm.insert('m', 128);
        hm.insert('u', 64);
        hm.insert('y', 32);
        hm.insert('l', 16);
        hm.insert('e', 4);
        hm.insert('t', 1);
        assert!(equal_hashmaps(
            &Bitap::pattern_alphabet("ward muylaert"),
            &hm
        ));
        let mut hm = HashMap::new();
        hm.insert('a', 1023);
        assert!(equal_hashmaps(&Bitap::pattern_alphabet("aaaaaaaaaa"), &hm));
    }

    #[test]
    fn test_bitap_hello_world() {
        let bitap = Bitap::new();
        let res = bitap.bitap("hello world", "elo");
        let mut indices = Vec::new();
        indices.push((1, 4));
        indices.push((7, 7));
        indices.push((9, 9));
        let goal = SearchResult {
            is_match: true,
            score: 0.343_333_333_333_333_3,
            matched_indices: indices,
        };
        assert_eq!(res, goal);
    }

    #[test]
    fn test_bitap_exact_hello_world() {
        let bitap = Bitap::new();
        let res = bitap.bitap("hello world", "hello");
        let mut indices = Vec::new();
        indices.push((0, 4));
        let goal = SearchResult {
            is_match: true,
            score: 0.001,
            matched_indices: indices,
        };
        assert_eq!(res, goal);
    }

    #[test]
    fn test_bitap_exact_hello_world2() {
        let bitap = Bitap::new();
        let res = bitap.bitap("hello world", "llo");
        let mut indices = Vec::new();
        indices.push((2, 4));
        let goal = SearchResult {
            is_match: true,
            score: 0.02,
            matched_indices: indices,
        };
        assert_eq!(res, goal);
    }

    #[test]
    fn test_pattern_longer_than_text() {
        // Mostly to see what happens then. Perhaps infuture hotwire a failure?
        // Otoh, "hello", "hello world" would still have some number...
        let bitap = Bitap::new();
        let res = bitap.bitap("a", "hello world");
        println!("{:#?}", res);
        // assert!(false);
    }

    #[test]
    fn test_search() {
        let bitap = Bitap::new();
        let mut haystack = vec![];
        haystack.push("hello world");
        haystack.push("Belgium");
        haystack.push("Running");
        haystack.push("Run away");
        haystack.push("ello mate");
        haystack.push("ward muylaert");
        dbg!(bitap.search("word", haystack));
        // panic!();
    }

    #[test]
    fn test_empty_query() {
        let bitap = Bitap::new();
        assert_eq!(
            bitap.bitap("hello world", ""),
            SearchResult {
                is_match: false,
                score: 1.0,
                matched_indices: vec![]
            }
        );
    }
}
