use std::collections::HashMap;

// Following https://github.com/krisk/Fuse/blob/master/src/bitap/bitap_search.js
// Probably should first do the supporting files in that folder
pub fn bitap(text: &str, pattern: &str) -> SearchResult {
    let textlength = text.len() as i64;
    let patternlength = pattern.len() as i64;
    let findallmatches = false;

    let pattern_alphabet = pattern_alphabet(pattern);

    // mask of the matches
    let mut match_mask = vec![0; textlength as usize];

    let expectedlocation = 0;
    let distance = 100; // What is this?
    let mut currentthreshold = 0.6;

    // TODO
    // best location check seems to be a speed up by using exact matching
    // skipping for now
    let mut best_location = -1;

    let mut last_bits: Vec<i64> = vec![0; text.len() + pattern.len() + 2]; // TODO just trying to make this work, is this ok?
    let mut finalscore = 1.0;
    let mut binmax = textlength + patternlength;

    let mask = 1 << (patternlength - 1);

    for i in 0..patternlength {
        let mut binmin = 0;
        let mut binmid = binmax;

        while binmin < binmid {
            let score = bitapscore(
                pattern,
                i,
                expectedlocation + binmid,
                expectedlocation,
                distance,
            );
            if score <= currentthreshold {
                binmin = binmid;
            } else {
                binmax = binmid;
            }
            binmid = (binmax - binmid) / 2 + binmin;
        }
        // result of the while becomes maximum for next iteration
        binmax = binmid;

        let mut start = std::cmp::max(1, expectedlocation - binmid + 1);
        let finish = if findallmatches {
            textlength
        } else {
            std::cmp::min(expectedlocation + binmid, textlength) + patternlength
        };
        dbg!(finish);

        // init bit array
        let mut bitarr = vec![0; (finish + 2) as usize];
        bitarr[finish as usize + 1] = (1 << i) - 1;
        dbg!(&bitarr);

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
            // println!(
            //     "bitarr len {}; last_bits len {}; j {}",
            //     bitarr.len(),
            //     last_bits.len(),
            //     j
            // );
            bitarr[j] = ((bitarr[j + 1] << 1) | 1) & charmatch;

            // subsequent passes fuzzy match
            if i != 0 {
                bitarr[j] |= (((last_bits[j + 1] | last_bits[j]) << 1) | 1) | last_bits[j + 1];
            }
            dbg!(bitarr[j]);

            if bitarr[j] & mask > 0 {
                finalscore = bitapscore(
                    pattern,
                    i,
                    currentlocation as i64,
                    expectedlocation,
                    distance,
                );

                if finalscore <= currentthreshold {
                    currentthreshold = finalscore;
                    best_location = currentlocation as i64;

                    if best_location <= expectedlocation {
                        break;
                    }

                    start = std::cmp::max(1, 2 * expectedlocation - best_location as i64);
                }
            }

            j -= 1;
        } // end while
        dbg!(expectedlocation);
        dbg!(i);

        let score = bitapscore(pattern, i + 1, expectedlocation, expectedlocation, distance);

        if score > currentthreshold {
            break;
        }

        last_bits = bitarr;
    }

    SearchResult {
        is_match: best_location >= 0,
        score: if finalscore == 0.0 { 0.001 } else { finalscore },
        matched_indices: matched_indices(match_mask, 1),
    }
}

#[derive(Debug)]
pub struct SearchResult {
    is_match: bool,
    score: f64,
    matched_indices: Vec<(i64, i64)>,
}

fn bitapscore(
    pattern: &str,
    errors: i64,
    currentlocation: i64,
    expectedlocation: i64,
    distance: i64,
) -> f64 {
    let accuracy = errors as f64 / pattern.len() as f64;
    let proximity = (expectedlocation as f64 - currentlocation as f64).abs();

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
    let mut end = -1;
    let mut i = 0;
    let matchmask_len = matchmask.len() as i64;

    while i < matchmask_len {
        let a_match = matchmask[i as usize];
        if a_match != 0 && start == -1 {
            start = i
        } else if a_match == 0 && start != -1 {
            end = i - 1;
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
        assert!(equal_hashmaps(&pattern_alphabet("hello"), &hm));
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
        assert!(equal_hashmaps(&pattern_alphabet("ward muylaert"), &hm));
        let mut hm = HashMap::new();
        hm.insert('a', 1023);
        assert!(equal_hashmaps(&pattern_alphabet("aaaaaaaaaa"), &hm));
    }

    #[test]
    fn test_bitap() {
        // For "hello world", "elo":
        // { isMatch: true,
        //   score: 0.3433333333333333,
        //   matchedIndices: [ [ 1, 4 ], [ 7, 7 ], [ 9, 9 ] ] }
        let res = bitap("hello world", "elo");
        let mut ok = true;
        ok = ok && res.is_match == true;
        ok = ok && res.score > 0.343 && res.score < 0.344;
        println!("{:#?}", res);
        assert!(ok);
        let mut indices = Vec::new();
        indices.push((1, 4));
        indices.push((7, 7));
        indices.push((9, 9));
        assert_eq!(indices, res.matched_indices);
    }
}
