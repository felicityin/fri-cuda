use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_field::{Field, extension::BinomialExtensionField};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{Hash, PaddingFreeSponge, TruncatedPermutation};

pub type F = KoalaBear;
pub type EF = BinomialExtensionField<F, 4>;
pub type Challenge = EF;

const WIDTH: usize = 16;
const RATE: usize = 8;
pub type Perm = Poseidon2KoalaBear<WIDTH>;
pub type Challenger = DuplexChallenger<F, Perm, WIDTH, RATE>;

const DIGEST_WIDTH: usize = 8;
pub type Digest = [F; DIGEST_WIDTH];
pub type DigestHash = Hash<F, F, DIGEST_WIDTH>;

pub type MyHash = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST_WIDTH>;
pub type MyCompress = TruncatedPermutation<Perm, 2, DIGEST_WIDTH, WIDTH>;
pub type ValMmcs =
    MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, DIGEST_WIDTH>;
pub type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
