use std::collections::HashMap;

// Following https://github.com/krisk/Fuse/blob/master/src/bitap/bitap_search.js
// Probably should first do the supporting files in that folder
pub fn bitap(text: &str, pattern: &str) {
    let textlength = text.len() as i64;
    let patternlength = pattern.len() as i64;

    // mask of the matches
    let mut match_mask = vec![textlength; 0];

    let expectedlocation = 0;
    let distance = 100; // What is this?
    let currentthreshold = 0.6;

    // best location check seems to be a speed up by using exact matching
    // skipping for now

    let best_location = -1;
    let last_bits: Vec<i64> = vec![];
    let finalscore = 1;
    let mut binmax = textlength + patternlength;

    let mask = 1 << (patternlength - 1);

    let mut binmin = 0;
    let mut binmid = binmax;
    for i in 0..patternlength {
        binmin = 0;
        binmid = binmax;

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

        let start = std::cmp::max(1, expectedlocation - binmid + 1);
        let findallmatches = true;
        let finish = if findallmatches {
            textlength
        } else {
            std::cmp::min(expectedlocation + binmid, textlength) + patternlength
        };

        // init bit array
        let mut bitarr = vec![finish + 2; 0];
        bitarr[finish as usize + 1] = (1 << i) - 1;

        let mut j = finish;
        while j >= start {
            let currentlocation = j - 1;
            //       let charMatch = patternAlphabet[text.charAt(currentLocation)]

            j -= 1;
        }
    }
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
            return 1.0;
        } else {
            return accuracy;
        }
    } else {
        return accuracy + (proximity / distance as f64);
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
        return true;
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
}
