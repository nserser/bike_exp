use anyhow::{bail, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SyndromeMask(u128);

impl SyndromeMask {
    pub fn empty() -> Self {
        Self(0)
    }

    pub fn set(&mut self, pos: usize) -> Result<()> {
        if pos >= 128 {
            bail!("syndrome bit index out of range: {pos}");
        }
        self.0 |= 1u128 << pos;
        Ok(())
    }

    pub fn xor_assign(&mut self, other: Self) {
        self.0 ^= other.0;
    }

    pub fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub fn popcount(self) -> u32 {
        self.0.count_ones()
    }

    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    pub fn contains(self, pos: usize) -> bool {
        pos < 128 && (self.0 & (1u128 << pos)) != 0
    }

    pub fn to_hex(self) -> String {
        format!("{:032x}", self.0)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mask256 {
    limbs: [u64; 4],
}

impl Mask256 {
    pub fn empty() -> Self {
        Self { limbs: [0; 4] }
    }

    pub fn set(&mut self, pos: usize) -> Result<()> {
        let (limb, bit) = Self::limb_and_bit(pos)?;
        self.limbs[limb] |= 1u64 << bit;
        Ok(())
    }

    pub fn clear(&mut self, pos: usize) -> Result<()> {
        let (limb, bit) = Self::limb_and_bit(pos)?;
        self.limbs[limb] &= !(1u64 << bit);
        Ok(())
    }

    pub fn contains(&self, pos: usize) -> bool {
        if pos >= 256 {
            return false;
        }
        let limb = pos / 64;
        let bit = pos % 64;
        (self.limbs[limb] & (1u64 << bit)) != 0
    }

    pub fn xor_assign(&mut self, other: Self) {
        for (lhs, rhs) in self.limbs.iter_mut().zip(other.limbs) {
            *lhs ^= rhs;
        }
    }

    pub fn union(self, other: Self) -> Self {
        let mut out = self;
        for (dst, rhs) in out.limbs.iter_mut().zip(other.limbs) {
            *dst |= rhs;
        }
        out
    }

    pub fn intersection(self, other: Self) -> Self {
        let mut out = Self::empty();
        for idx in 0..4 {
            out.limbs[idx] = self.limbs[idx] & other.limbs[idx];
        }
        out
    }

    pub fn popcount(&self) -> u32 {
        self.limbs.iter().map(|v| v.count_ones()).sum()
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|v| *v == 0)
    }

    pub fn trim_unused(&mut self, max_bits: usize) -> Result<()> {
        if max_bits > 256 {
            bail!("max_bits must be <= 256");
        }
        if max_bits == 256 {
            return Ok(());
        }

        let full_limbs = max_bits / 64;
        let rem_bits = max_bits % 64;
        for limb in self.limbs.iter_mut().skip(full_limbs + usize::from(rem_bits > 0)) {
            *limb = 0;
        }
        if rem_bits > 0 {
            self.limbs[full_limbs] &= (1u64 << rem_bits) - 1;
        }
        Ok(())
    }

    pub fn iter_ones(&self, max_bits: usize) -> impl Iterator<Item = usize> + '_ {
        let limit = max_bits.min(256);
        self.limbs.iter().enumerate().flat_map(move |(limb_idx, limb)| {
            let start = limb_idx * 64;
            let end = (start + 64).min(limit);
            (start..end).filter(move |pos| (limb & (1u64 << (pos - start))) != 0)
        })
    }

    pub fn to_hex(&self) -> String {
        self.limbs
            .iter()
            .rev()
            .map(|limb| format!("{limb:016x}"))
            .collect::<String>()
    }

    fn limb_and_bit(pos: usize) -> Result<(usize, usize)> {
        if pos >= 256 {
            bail!("bit index out of range: {pos}");
        }
        Ok((pos / 64, pos % 64))
    }
}

#[cfg(test)]
mod tests {
    use super::{Mask256, SyndromeMask};

    #[test]
    fn syndrome_mask_basic_operations() {
        let mut mask = SyndromeMask::empty();
        assert!(mask.is_zero());
        mask.set(0).unwrap();
        mask.set(127).unwrap();
        assert!(mask.contains(0));
        assert!(mask.contains(127));
        assert_eq!(mask.popcount(), 2);
        assert_eq!(mask.to_hex(), "80000000000000000000000000000001");

        let mut other = SyndromeMask::empty();
        other.set(127).unwrap();
        mask.xor_assign(other);
        assert!(mask.contains(0));
        assert!(!mask.contains(127));
        assert_eq!(mask.and(other).popcount(), 0);
    }

    #[test]
    fn syndrome_mask_rejects_out_of_range_bits() {
        let mut mask = SyndromeMask::empty();
        assert!(mask.set(128).is_err());
    }

    #[test]
    fn mask256_tracks_set_bits_and_hex() {
        let mut mask = Mask256::empty();
        mask.set(0).unwrap();
        mask.set(63).unwrap();
        mask.set(64).unwrap();
        mask.set(255).unwrap();
        assert!(mask.contains(0));
        assert!(mask.contains(255));
        assert_eq!(mask.popcount(), 4);
        assert_eq!(
            mask.to_hex(),
            "8000000000000000000000000000000000000000000000018000000000000001"
        );
    }

    #[test]
    fn mask256_union_intersection_and_trim() {
        let mut lhs = Mask256::empty();
        lhs.set(1).unwrap();
        lhs.set(70).unwrap();

        let mut rhs = Mask256::empty();
        rhs.set(70).unwrap();
        rhs.set(130).unwrap();

        assert_eq!(lhs.intersection(rhs).popcount(), 1);
        assert_eq!(lhs.union(rhs).popcount(), 3);

        let mut trimmed = rhs.union(lhs);
        trimmed.trim_unused(128).unwrap();
        assert!(!trimmed.contains(130));
        assert_eq!(trimmed.popcount(), 2);
    }

    #[test]
    fn mask256_iter_ones_respects_limit() {
        let mut mask = Mask256::empty();
        mask.set(1).unwrap();
        mask.set(70).unwrap();
        mask.set(130).unwrap();
        let bits = mask.iter_ones(128).collect::<Vec<_>>();
        assert_eq!(bits, vec![1, 70]);
    }
}
