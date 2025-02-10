use crate::DAP_VERSION_IDENTIFIER;
use janus_messages::{taskprov, TaskId};
use prio::{
    field::Field64,
    flp::{
        gadgets::{Mul, ParallelSumGadget},
        types::SumVec,
    },
    vdaf::{prio3::Prio3, xof::XofHmacSha256Aes128, VdafError},
};
use serde::{Deserialize, Serialize};
use std::str;

/// The length of the verify key parameter for Prio3 VDAF instantiations using
/// [`XofTurboShake128`][prio::vdaf::xof::XofTurboShake128].
pub const VERIFY_KEY_LENGTH_PRIO3: usize = 32;

/// The length of the verify key parameter when using [`XofHmacSha256Aes128`]. This XOF is not part
/// of the VDAF specification.
pub const VERIFY_KEY_LENGTH_PRIO3_HMACSHA256_AES128: usize = 32;

/// Private use algorithm ID for a customized version of Prio3SumVec. This value was chosen for
/// interoperability with Daphne.
const ALGORITHM_ID_PRIO3_SUM_VEC_FIELD64_MULTIPROOF_HMACSHA256_AES128: u32 = 0xFFFF_1003;

/// Computes the VDAF "application context", which is an opaque byte string used by many VDAF
/// operations for domain-separation purposes. In DAP, this is the DAP draft number concatenated
/// with the task ID.
pub fn vdaf_application_context(
    task_id: &TaskId,
) -> [u8; DAP_VERSION_IDENTIFIER.len() + TaskId::LEN] {
    let mut ctx = [0u8; DAP_VERSION_IDENTIFIER.len() + TaskId::LEN];
    ctx[..DAP_VERSION_IDENTIFIER.len()].copy_from_slice(DAP_VERSION_IDENTIFIER.as_bytes());
    ctx[DAP_VERSION_IDENTIFIER.len()..].copy_from_slice(task_id.as_ref());
    ctx
}

/// Bitsize parameter for the `Prio3FixedPointBoundedL2VecSum` VDAF.
#[cfg(feature = "fpvec_bounded_l2")]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Prio3FixedPointBoundedL2VecSumBitSize {
    BitSize16,
    BitSize32,
}

/// Contains dedicated enums which describe the differential privacy strategies
/// of a given VDAF.
///
/// If a VDAF only supports a single strategy, such as for example `NoDifferentialPrivacy`, then no
/// enum is required.
pub mod vdaf_dp_strategies {
    use prio::dp::distributions::PureDpDiscreteLaplace;
    #[cfg(feature = "fpvec_bounded_l2")]
    use prio::dp::distributions::ZCdpDiscreteGaussian;
    use serde::{Deserialize, Serialize};

    /// Differential privacy strategies supported by `Prio3Histogram`.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    #[serde(tag = "dp_strategy")]
    pub enum Prio3Histogram {
        NoDifferentialPrivacy,
        PureDpDiscreteLaplace(PureDpDiscreteLaplace),
    }

    impl Default for Prio3Histogram {
        fn default() -> Self {
            Self::NoDifferentialPrivacy
        }
    }

    /// Differential privacy strategies supported by `Prio3SumVec`.
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    #[serde(tag = "dp_strategy")]
    pub enum Prio3SumVec {
        NoDifferentialPrivacy,
        PureDpDiscreteLaplace(PureDpDiscreteLaplace),
    }

    impl Default for Prio3SumVec {
        fn default() -> Self {
            Self::NoDifferentialPrivacy
        }
    }

    /// Differential privacy strategies supported by `Prio3FixedPointBoundedL2VecSum`.
    #[cfg(feature = "fpvec_bounded_l2")]
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    #[serde(tag = "dp_strategy")]
    pub enum Prio3FixedPointBoundedL2VecSum {
        NoDifferentialPrivacy,
        ZCdpDiscreteGaussian(ZCdpDiscreteGaussian),
    }

    #[cfg(feature = "fpvec_bounded_l2")]
    impl Default for Prio3FixedPointBoundedL2VecSum {
        fn default() -> Self {
            Self::NoDifferentialPrivacy
        }
    }
}

/// Identifiers for supported VDAFs, corresponding to definitions in
/// [draft-irtf-cfrg-vdaf-03][1] and implementations in [`prio::vdaf::prio3`].
///
/// [1]: https://datatracker.ietf.org/doc/draft-irtf-cfrg-vdaf/03/
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum VdafInstance {
    /// A `Prio3` counter.
    Prio3Count,
    /// A `Prio3` sum.
    Prio3Sum { max_measurement: u64 },
    /// A vector of `Prio3` sums.
    Prio3SumVec {
        bits: usize,
        length: usize,
        chunk_length: usize,
        #[serde(default)]
        dp_strategy: vdaf_dp_strategies::Prio3SumVec,
    },
    /// Prio3SumVec with additional customizations: a smaller field, multiple proofs, and a
    /// different XOF.
    Prio3SumVecField64MultiproofHmacSha256Aes128 {
        proofs: u8,
        bits: usize,
        length: usize,
        chunk_length: usize,
        #[serde(default)]
        dp_strategy: vdaf_dp_strategies::Prio3SumVec,
    },
    /// A `Prio3` histogram with `length` buckets in it.
    Prio3Histogram {
        length: usize,
        chunk_length: usize,
        #[serde(default)]
        dp_strategy: vdaf_dp_strategies::Prio3Histogram,
    },
    /// A `Prio3` fixed point vector sum with bounded L2 norm.
    #[cfg(feature = "fpvec_bounded_l2")]
    Prio3FixedPointBoundedL2VecSum {
        bitsize: Prio3FixedPointBoundedL2VecSumBitSize,
        dp_strategy: vdaf_dp_strategies::Prio3FixedPointBoundedL2VecSum,
        length: usize,
    },

    /// A fake, no-op VDAF, which uses an aggregation parameter and a variable number of rounds.
    #[cfg(feature = "test-util")]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
    Fake { rounds: u32 },
    /// A fake, no-op VDAF that always fails during initialization of input preparation.
    #[cfg(feature = "test-util")]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
    FakeFailsPrepInit,
    /// A fake, no-op VDAF that always fails when stepping input preparation.
    #[cfg(feature = "test-util")]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
    FakeFailsPrepStep,
}

impl VdafInstance {
    /// Returns the expected length of a VDAF verification key for a VDAF of this type.
    pub fn verify_key_length(&self) -> usize {
        match self {
            #[cfg(feature = "test-util")]
            VdafInstance::Fake { .. }
            | VdafInstance::FakeFailsPrepInit
            | VdafInstance::FakeFailsPrepStep => 0,

            VdafInstance::Prio3SumVecField64MultiproofHmacSha256Aes128 { .. } => {
                VERIFY_KEY_LENGTH_PRIO3_HMACSHA256_AES128
            }

            // All other VDAFs (Prio3 as-specified) have the same verify key length.
            _ => VERIFY_KEY_LENGTH_PRIO3,
        }
    }
}

impl TryFrom<&taskprov::VdafConfig> for VdafInstance {
    type Error = &'static str;

    fn try_from(value: &taskprov::VdafConfig) -> Result<Self, Self::Error> {
        match value {
            taskprov::VdafConfig::Prio3Count => Ok(Self::Prio3Count),
            taskprov::VdafConfig::Prio3Sum { max_measurement } => Ok(Self::Prio3Sum {
                max_measurement: u64::from(*max_measurement),
            }),
            taskprov::VdafConfig::Prio3SumVec {
                bits,
                length,
                chunk_length,
            } => Ok(Self::Prio3SumVec {
                bits: *bits as usize,
                length: *length as usize,
                chunk_length: *chunk_length as usize,
                dp_strategy: vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy,
            }),
            taskprov::VdafConfig::Prio3SumVecField64MultiproofHmacSha256Aes128 {
                bits,
                length,
                chunk_length,
                proofs,
            } => Ok(Self::Prio3SumVecField64MultiproofHmacSha256Aes128 {
                proofs: *proofs,
                bits: *bits as usize,
                length: *length as usize,
                chunk_length: *chunk_length as usize,
                dp_strategy: vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy,
            }),
            taskprov::VdafConfig::Prio3Histogram {
                length,
                chunk_length,
            } => Ok(Self::Prio3Histogram {
                length: *length as usize,
                chunk_length: *chunk_length as usize,
                dp_strategy: vdaf_dp_strategies::Prio3Histogram::NoDifferentialPrivacy,
            }),

            #[cfg(feature = "test-util")]
            taskprov::VdafConfig::Fake { rounds } => Ok(Self::Fake { rounds: *rounds }),

            _ => Err("unknown VdafType"),
        }
    }
}

pub type Prio3SumVecField64MultiproofHmacSha256Aes128<PS> =
    Prio3<SumVec<Field64, PS>, XofHmacSha256Aes128, 32>;

/// Construct a customized Prio3SumVec VDAF, using the [`Field64`] field, multiple proofs, and
/// [`XofHmacSha256Aes128`] as the XOF.
pub fn new_prio3_sum_vec_field64_multiproof_hmacsha256_aes128<
    PS: ParallelSumGadget<Field64, Mul<Field64>> + Eq + 'static,
>(
    proofs: u8,
    bits: usize,
    length: usize,
    chunk_length: usize,
) -> Result<Prio3SumVecField64MultiproofHmacSha256Aes128<PS>, VdafError> {
    if proofs < 2 {
        return Err(VdafError::Uncategorized(
            "Must use at least two proofs with Field64".into(),
        ));
    }
    Prio3::new(
        2,
        proofs,
        ALGORITHM_ID_PRIO3_SUM_VEC_FIELD64_MULTIPROOF_HMACSHA256_AES128,
        SumVec::new(bits, length, chunk_length)?,
    )
}

/// Internal implementation details of [`vdaf_dispatch`](crate::vdaf_dispatch).
#[macro_export]
macro_rules! vdaf_dispatch_impl_base {
    (impl match base $vdaf_instance:expr, ($vdaf:ident, $Vdaf:ident, $VERIFY_KEY_LEN:ident, $dp_strategy:ident, $DpStrategy:ident) => $body:tt) => {
        match $vdaf_instance {
            ::janus_core::vdaf::VdafInstance::Prio3Count => {
                let $vdaf = ::prio::vdaf::prio3::Prio3::new_count(2)?;
                type $Vdaf = ::prio::vdaf::prio3::Prio3Count;
                const $VERIFY_KEY_LEN: usize = ::janus_core::vdaf::VERIFY_KEY_LENGTH_PRIO3;
                type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                let $dp_strategy = janus_core::dp::NoDifferentialPrivacy;
                $body
            }

            ::janus_core::vdaf::VdafInstance::Prio3Sum { max_measurement } => {
                let $vdaf = ::prio::vdaf::prio3::Prio3::new_sum(2, *max_measurement)?;
                type $Vdaf = ::prio::vdaf::prio3::Prio3Sum;
                const $VERIFY_KEY_LEN: usize = ::janus_core::vdaf::VERIFY_KEY_LENGTH_PRIO3;
                type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                let $dp_strategy = janus_core::dp::NoDifferentialPrivacy;
                $body
            }

            ::janus_core::vdaf::VdafInstance::Prio3SumVec {
                bits,
                length,
                chunk_length,
                dp_strategy,
            } => {
                let $vdaf =
                    ::prio::vdaf::prio3::Prio3::new_sum_vec(2, *bits, *length, *chunk_length)?;
                type $Vdaf = ::prio::vdaf::prio3::Prio3SumVec;
                const $VERIFY_KEY_LEN: usize = ::janus_core::vdaf::VERIFY_KEY_LENGTH_PRIO3;
                match dp_strategy.clone() {
                    ::janus_core::vdaf::vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy => {
                        type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                        let $dp_strategy = janus_core::dp::NoDifferentialPrivacy;
                        $body
                    }
                    ::janus_core::vdaf::vdaf_dp_strategies::Prio3SumVec::PureDpDiscreteLaplace(
                        _strategy,
                    ) => {
                        type $DpStrategy = ::prio::dp::distributions::PureDpDiscreteLaplace;
                        let $dp_strategy = _strategy;
                        $body
                    }
                }
            }

            ::janus_core::vdaf::VdafInstance::Prio3SumVecField64MultiproofHmacSha256Aes128 {
                proofs,
                bits,
                length,
                chunk_length,
                dp_strategy,
            } => {
                let $vdaf =
                    janus_core::vdaf::new_prio3_sum_vec_field64_multiproof_hmacsha256_aes128(
                        *proofs,
                        *bits,
                        *length,
                        *chunk_length,
                    )?;
                type $Vdaf = janus_core::vdaf::Prio3SumVecField64MultiproofHmacSha256Aes128<
                    ::prio::flp::gadgets::ParallelSum<
                        ::prio::field::Field64,
                        ::prio::flp::gadgets::Mul<::prio::field::Field64>,
                    >,
                >;
                const $VERIFY_KEY_LEN: usize =
                    ::janus_core::vdaf::VERIFY_KEY_LENGTH_PRIO3_HMACSHA256_AES128;
                match dp_strategy.clone() {
                    ::janus_core::vdaf::vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy => {
                        type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                        let $dp_strategy = janus_core::dp::NoDifferentialPrivacy;
                        $body
                    }
                    ::janus_core::vdaf::vdaf_dp_strategies::Prio3SumVec::PureDpDiscreteLaplace(
                        _strategy,
                    ) => {
                        type $DpStrategy = ::prio::dp::distributions::PureDpDiscreteLaplace;
                        let $dp_strategy = _strategy;
                        $body
                    }
                }
            }

            ::janus_core::vdaf::VdafInstance::Prio3Histogram {
                length,
                chunk_length,
                dp_strategy,
            } => {
                let $vdaf = ::prio::vdaf::prio3::Prio3::new_histogram(2, *length, *chunk_length)?;
                type $Vdaf = ::prio::vdaf::prio3::Prio3Histogram;
                const $VERIFY_KEY_LEN: usize = ::janus_core::vdaf::VERIFY_KEY_LENGTH_PRIO3;
                match dp_strategy.clone() {
                    ::janus_core::vdaf::vdaf_dp_strategies::Prio3Histogram::NoDifferentialPrivacy => {
                        type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                        let $dp_strategy = janus_core::dp::NoDifferentialPrivacy;
                        $body
                    }
                    ::janus_core::vdaf::vdaf_dp_strategies::Prio3Histogram::PureDpDiscreteLaplace(_strategy) => {
                        type $DpStrategy = ::prio::dp::distributions::PureDpDiscreteLaplace;
                        let $dp_strategy = _strategy;
                        $body
                    }
                }
            }

            _ => unreachable!(),
        }
    };
}

/// Internal implementation details of [`vdaf_dispatch`](crate::vdaf_dispatch).
#[cfg(feature = "fpvec_bounded_l2")]
#[macro_export]
macro_rules! vdaf_dispatch_impl_fpvec_bounded_l2 {
    (impl match fpvec_bounded_l2 $vdaf_instance:expr, ($vdaf:ident, $Vdaf:ident, $VERIFY_KEY_LEN:ident, $dp_strategy:ident, $DpStrategy:ident) => $body:tt) => {
        match $vdaf_instance {
            ::janus_core::vdaf::VdafInstance::Prio3FixedPointBoundedL2VecSum { bitsize, dp_strategy, length } => {
                const $VERIFY_KEY_LEN: usize = ::janus_core::vdaf::VERIFY_KEY_LENGTH_PRIO3;

                match dp_strategy.clone() {
                    janus_core::vdaf::vdaf_dp_strategies::Prio3FixedPointBoundedL2VecSum::ZCdpDiscreteGaussian(_strategy) => {
                        type $DpStrategy = ::prio::dp::distributions::ZCdpDiscreteGaussian;
                        let $dp_strategy = _strategy;
                        janus_core::vdaf_dispatch_impl_fpvec_bounded_l2!(@dispatch_bitsize bitsize, $Vdaf, $vdaf, length => $body)
                    },
                    janus_core::vdaf::vdaf_dp_strategies::Prio3FixedPointBoundedL2VecSum::NoDifferentialPrivacy => {
                        type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                        let $dp_strategy = janus_core::dp::NoDifferentialPrivacy;
                        janus_core::vdaf_dispatch_impl_fpvec_bounded_l2!(@dispatch_bitsize bitsize, $Vdaf, $vdaf, length => $body)
                    }
                }
            },

            _ => unreachable!(),
        }
    };

    (@dispatch_bitsize $bitsize:ident, $Vdaf:ident, $vdaf:ident, $length:ident => $body:tt) => {
        match $bitsize {
            janus_core::vdaf::Prio3FixedPointBoundedL2VecSumBitSize::BitSize16 => {
                let $vdaf =
                    ::prio::vdaf::prio3::Prio3::new_fixedpoint_boundedl2_vec_sum(
                        2, *$length,
                    )?;
                type $Vdaf = ::prio::vdaf::prio3::Prio3FixedPointBoundedL2VecSum<
                    ::fixed::FixedI16<::fixed::types::extra::U15>,
                >;
                $body
            },
            janus_core::vdaf::Prio3FixedPointBoundedL2VecSumBitSize::BitSize32 => {
                let $vdaf =
                    ::prio::vdaf::prio3::Prio3::new_fixedpoint_boundedl2_vec_sum(
                        2, *$length,
                    )?;
                type $Vdaf = ::prio::vdaf::prio3::Prio3FixedPointBoundedL2VecSum<
                    ::fixed::FixedI32<::fixed::types::extra::U31>,
                >;
                $body
            },
        };
    }
}

/// Internal implementation details of [`vdaf_dispatch`](crate::vdaf_dispatch).
#[cfg(feature = "test-util")]
#[macro_export]
macro_rules! vdaf_dispatch_impl_test_util {
    (impl match test_util $vdaf_instance:expr, ($vdaf:ident, $Vdaf:ident, $VERIFY_KEY_LEN:ident, $dp_strategy:ident, $DpStrategy:ident) => $body:tt) => {
        match $vdaf_instance {
            ::janus_core::vdaf::VdafInstance::Fake { rounds } => {
                let $vdaf = ::prio::vdaf::dummy::Vdaf::new(*rounds);
                type $Vdaf = ::prio::vdaf::dummy::Vdaf;
                const $VERIFY_KEY_LEN: usize = 0;
                type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                let $dp_strategy = janus_core::dp::NoDifferentialPrivacy;
                $body
            }

            ::janus_core::vdaf::VdafInstance::FakeFailsPrepInit => {
                let $vdaf = ::prio::vdaf::dummy::Vdaf::new(1).with_prep_init_fn(
                    |_| -> Result<(), ::prio::vdaf::VdafError> {
                        ::std::result::Result::Err(::prio::vdaf::VdafError::Uncategorized(
                            "FakeFailsPrepInit failed at prep_init".to_string(),
                        ))
                    },
                );
                type $Vdaf = ::prio::vdaf::dummy::Vdaf;
                const $VERIFY_KEY_LEN: usize = 0;
                type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                let $dp_strategy = janus_core::dp::NoDifferentialPrivacy;
                $body
            }

            ::janus_core::vdaf::VdafInstance::FakeFailsPrepStep => {
                let $vdaf = ::prio::vdaf::dummy::Vdaf::new(1).with_prep_step_fn(
                    |_| -> Result<
                        ::prio::vdaf::PrepareTransition<::prio::vdaf::dummy::Vdaf, 0, 16>,
                        ::prio::vdaf::VdafError,
                    > {
                        ::std::result::Result::Err(::prio::vdaf::VdafError::Uncategorized(
                            "FakeFailsPrepStep failed at prep_step".to_string(),
                        ))
                    },
                );
                type $Vdaf = ::prio::vdaf::dummy::Vdaf;
                const $VERIFY_KEY_LEN: usize = 0;
                type $DpStrategy = janus_core::dp::NoDifferentialPrivacy;
                let $dp_strategy = janus_core::dp::NoDifferentialPrivacy;
                $body
            }

            _ => unreachable!(),
        }
    };
}

/// Internal implementation details of [`vdaf_dispatch`](crate::vdaf_dispatch).
#[cfg(all(feature = "fpvec_bounded_l2", feature = "test-util"))]
#[macro_export]
macro_rules! vdaf_dispatch_impl {
    (impl match all $vdaf_instance:expr, ($vdaf:ident, $Vdaf:ident, $VERIFY_KEY_LEN:ident, $dp_strategy:ident, $DpStrategy:ident) => $body:tt) => {
        match $vdaf_instance {
            ::janus_core::vdaf::VdafInstance::Prio3Count
            | ::janus_core::vdaf::VdafInstance::Prio3Sum { .. }
            | ::janus_core::vdaf::VdafInstance::Prio3SumVec { .. }
            | ::janus_core::vdaf::VdafInstance::Prio3SumVecField64MultiproofHmacSha256Aes128 { .. }
            | ::janus_core::vdaf::VdafInstance::Prio3Histogram { .. } => {
                ::janus_core::vdaf_dispatch_impl_base!(impl match base $vdaf_instance, ($vdaf, $Vdaf, $VERIFY_KEY_LEN, $dp_strategy, $DpStrategy) => $body)
            }

            ::janus_core::vdaf::VdafInstance::Prio3FixedPointBoundedL2VecSum { .. } => {
                ::janus_core::vdaf_dispatch_impl_fpvec_bounded_l2!(impl match fpvec_bounded_l2 $vdaf_instance, ($vdaf, $Vdaf, $VERIFY_KEY_LEN, $dp_strategy, $DpStrategy) => $body)
            }

            ::janus_core::vdaf::VdafInstance::Fake { .. }
            | ::janus_core::vdaf::VdafInstance::FakeFailsPrepInit
            | ::janus_core::vdaf::VdafInstance::FakeFailsPrepStep => {
                ::janus_core::vdaf_dispatch_impl_test_util!(impl match test_util $vdaf_instance, ($vdaf, $Vdaf, $VERIFY_KEY_LEN, $dp_strategy, $DpStrategy) => $body)
            }

            _ => panic!("VDAF {:?} is not yet supported", $vdaf_instance),
        }
    };
}

/// Internal implementation details of [`vdaf_dispatch`](crate::vdaf_dispatch).
#[cfg(all(feature = "fpvec_bounded_l2", not(feature = "test-util")))]
#[macro_export]
macro_rules! vdaf_dispatch_impl {
    (impl match all $vdaf_instance:expr, ($vdaf:ident, $Vdaf:ident, $VERIFY_KEY_LEN:ident, $dp_strategy:ident, $DpStrategy:ident) => $body:tt) => {
        match $vdaf_instance {
            ::janus_core::vdaf::VdafInstance::Prio3Count
            | ::janus_core::vdaf::VdafInstance::Prio3Sum { .. }
            | ::janus_core::vdaf::VdafInstance::Prio3SumVec { .. }
            | ::janus_core::vdaf::VdafInstance::Prio3SumVecField64MultiproofHmacSha256Aes128 { .. }
            | ::janus_core::vdaf::VdafInstance::Prio3Histogram { .. } => {
                ::janus_core::vdaf_dispatch_impl_base!(impl match base $vdaf_instance, ($vdaf, $Vdaf, $VERIFY_KEY_LEN, $dp_strategy, $DpStrategy) => $body)
            }

            ::janus_core::vdaf::VdafInstance::Prio3FixedPointBoundedL2VecSum { .. } => {
                ::janus_core::vdaf_dispatch_impl_fpvec_bounded_l2!(impl match fpvec_bounded_l2 $vdaf_instance, ($vdaf, $Vdaf, $VERIFY_KEY_LEN, $dp_strategy, $DpStrategy) => $body)
            }

            _ => panic!("VDAF {:?} is not yet supported", $vdaf_instance),
        }
    };
}

/// Internal implementation details of [`vdaf_dispatch`](crate::vdaf_dispatch).
#[cfg(all(not(feature = "fpvec_bounded_l2"), feature = "test-util"))]
#[macro_export]
macro_rules! vdaf_dispatch_impl {
    (impl match all $vdaf_instance:expr, ($vdaf:ident, $Vdaf:ident, $VERIFY_KEY_LEN:ident, $dp_strategy:ident, $DpStrategy:ident) => $body:tt) => {
        match $vdaf_instance {
            ::janus_core::vdaf::VdafInstance::Prio3Count
            | ::janus_core::vdaf::VdafInstance::Prio3Sum { .. }
            | ::janus_core::vdaf::VdafInstance::Prio3SumVec { .. }
            | ::janus_core::vdaf::VdafInstance::Prio3SumVecField64MultiproofHmacSha256Aes128 { .. }
            | ::janus_core::vdaf::VdafInstance::Prio3Histogram { .. } => {
                ::janus_core::vdaf_dispatch_impl_base!(impl match base $vdaf_instance, ($vdaf, $Vdaf, $VERIFY_KEY_LEN, $dp_strategy, $DpStrategy) => $body)
            }

            ::janus_core::vdaf::VdafInstance::Fake { .. }
            | ::janus_core::vdaf::VdafInstance::FakeFailsPrepInit
            | ::janus_core::vdaf::VdafInstance::FakeFailsPrepStep => {
                ::janus_core::vdaf_dispatch_impl_test_util!(impl match test_util $vdaf_instance, ($vdaf, $Vdaf, $VERIFY_KEY_LEN, $dp_strategy, $DpStrategy) => $body)
            }

            _ => panic!("VDAF {:?} is not yet supported", $vdaf_instance),
        }
    };
}

/// Internal implementation details of [`vdaf_dispatch`](crate::vdaf_dispatch).
#[cfg(all(not(feature = "fpvec_bounded_l2"), not(feature = "test-util")))]
#[macro_export]
macro_rules! vdaf_dispatch_impl {
    (impl match all $vdaf_instance:expr, ($vdaf:ident, $Vdaf:ident, $VERIFY_KEY_LEN:ident, $dp_strategy:ident, $DpStrategy:ident) => $body:tt) => {
        match $vdaf_instance {
            ::janus_core::vdaf::VdafInstance::Prio3Count
            | ::janus_core::vdaf::VdafInstance::Prio3Sum { .. }
            | ::janus_core::vdaf::VdafInstance::Prio3SumVec { .. }
            | ::janus_core::vdaf::VdafInstance::Prio3SumVecField64MultiproofHmacSha256Aes128 { .. }
            | ::janus_core::vdaf::VdafInstance::Prio3Histogram { .. } => {
                ::janus_core::vdaf_dispatch_impl_base!(impl match base $vdaf_instance, ($vdaf, $Vdaf, $VERIFY_KEY_LEN, $dp_strategy, $DpStrategy) => $body)
            }

            _ => panic!("VDAF {:?} is not yet supported", $vdaf_instance),
        }
    };
}

/// Emits a match block dispatching on a [`VdafInstance`].
///
/// This must be called inside a method that returns a result, with an error type that
/// [`prio::vdaf::VdafError`] can be converted into. Takes a `&VdafInstance` as the first argument,
/// followed by a pseudo-pattern and body. The pseudo-pattern takes a variable name for the
/// constructed VDAF, a type alias name that the block can use to explicitly specify the VDAF's
/// type, and the name of a const that will be set to the VDAF's verify key length, also for
/// explicitly specifying type parameters.
///
/// # Example:
///
/// ```
/// # use janus_core::vdaf_dispatch;
/// # fn handle_request_generic<A, const SEED_SIZE: usize>(_vdaf: &A) -> Result<(), prio::vdaf::VdafError>
/// # where
/// #     A: prio::vdaf::Aggregator<SEED_SIZE, 16>,
/// # {
/// #     Ok(())
/// # }
/// # fn test() -> Result<(), prio::vdaf::VdafError> {
/// #     let vdaf = janus_core::vdaf::VdafInstance::Prio3Count;
/// vdaf_dispatch!(&vdaf, (vdaf, VdafType, VERIFY_KEY_LEN) => {
///     handle_request_generic::<VdafType, VERIFY_KEY_LEN>(&vdaf)
/// })
/// # }
/// ```
#[macro_export]
macro_rules! vdaf_dispatch {
    // Provide the dispatched type only, don't construct a VDAF instance.
    ($vdaf_instance:expr, (_, $Vdaf:ident, $VERIFY_KEY_LEN:ident) => $body:tt) => {
        ::janus_core::vdaf_dispatch_impl!(impl match all $vdaf_instance, (_, $Vdaf, $VERIFY_KEY_LEN) => $body)
    };

    // Construct a VDAF instance, and provide that to the block as well.
    ($vdaf_instance:expr, ($vdaf:ident, $Vdaf:ident, $VERIFY_KEY_LEN:ident) => $body:tt) => {
        ::janus_core::vdaf_dispatch_impl!(impl match all $vdaf_instance, ($vdaf, $Vdaf, $VERIFY_KEY_LEN, _unused, _Unused) => $body)
    };

    // Construct a VDAF instance and DP strategy, and provide them to the block as well.
    ($vdaf_instance:expr, ($vdaf:ident, $Vdaf:ident, $VERIFY_KEY_LEN:ident, $dp_strategy:ident, $DpStrategy:ident) => $body:tt) => {
        ::janus_core::vdaf_dispatch_impl!(impl match all $vdaf_instance, ($vdaf, $Vdaf, $VERIFY_KEY_LEN, $dp_strategy, $DpStrategy) => $body)
    };
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "fpvec_bounded_l2")]
    use crate::vdaf::Prio3FixedPointBoundedL2VecSumBitSize;
    use crate::vdaf::{vdaf_dp_strategies, VdafInstance};
    use assert_matches::assert_matches;
    #[cfg(feature = "fpvec_bounded_l2")]
    use prio::dp::{distributions::ZCdpDiscreteGaussian, ZCdpBudget};
    use prio::dp::{
        distributions::{DiscreteLaplaceDpStrategy, PureDpDiscreteLaplace},
        DifferentialPrivacyStrategy, PureDpBudget, Rational,
    };
    use serde_test::{assert_tokens, Token};

    #[test]
    fn vdaf_serialization() {
        // The `Vdaf` type must have a stable serialization, as it gets stored in a JSON database
        // column.
        assert_tokens(
            &VdafInstance::Prio3Count,
            &[Token::UnitVariant {
                name: "VdafInstance",
                variant: "Prio3Count",
            }],
        );
        assert_tokens(
            &VdafInstance::Prio3Sum {
                max_measurement: 4096,
            },
            &[
                Token::StructVariant {
                    name: "VdafInstance",
                    variant: "Prio3Sum",
                    len: 1,
                },
                Token::Str("max_measurement"),
                Token::U64(4096),
                Token::StructVariantEnd,
            ],
        );
        assert_tokens(
            &VdafInstance::Prio3SumVec {
                bits: 1,
                length: 8,
                chunk_length: 3,
                dp_strategy: vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy,
            },
            &[
                Token::StructVariant {
                    name: "VdafInstance",
                    variant: "Prio3SumVec",
                    len: 4,
                },
                Token::Str("bits"),
                Token::U64(1),
                Token::Str("length"),
                Token::U64(8),
                Token::Str("chunk_length"),
                Token::U64(3),
                Token::Str("dp_strategy"),
                Token::Struct {
                    name: "Prio3SumVec",
                    len: 1,
                },
                Token::Str("dp_strategy"),
                Token::Str("NoDifferentialPrivacy"),
                Token::StructEnd,
                Token::StructVariantEnd,
            ],
        );
        assert_tokens(
            &VdafInstance::Prio3SumVec {
                bits: 1,
                length: 8,
                chunk_length: 3,
                dp_strategy: vdaf_dp_strategies::Prio3SumVec::PureDpDiscreteLaplace(
                    PureDpDiscreteLaplace::from_budget(
                        PureDpBudget::new(Rational::from_unsigned(2u128, 1u128).unwrap()).unwrap(),
                    ),
                ),
            },
            &[
                Token::StructVariant {
                    name: "VdafInstance",
                    variant: "Prio3SumVec",
                    len: 4,
                },
                Token::Str("bits"),
                Token::U64(1),
                Token::Str("length"),
                Token::U64(8),
                Token::Str("chunk_length"),
                Token::U64(3),
                Token::Str("dp_strategy"),
                Token::Struct {
                    name: "DiscreteLaplaceDpStrategy",
                    len: 2,
                },
                Token::Str("dp_strategy"),
                Token::Str("PureDpDiscreteLaplace"),
                Token::Str("budget"),
                Token::Struct {
                    name: "PureDpBudget",
                    len: 1,
                },
                Token::Str("epsilon"),
                Token::Tuple { len: 2 },
                Token::Seq { len: Some(1) },
                Token::U32(2),
                Token::SeqEnd,
                Token::Seq { len: Some(1) },
                Token::U32(1),
                Token::SeqEnd,
                Token::TupleEnd,
                Token::StructEnd,
                Token::StructEnd,
                Token::StructVariantEnd,
            ],
        );
        assert_tokens(
            &VdafInstance::Prio3SumVecField64MultiproofHmacSha256Aes128 {
                proofs: 2,
                bits: 1,
                length: 8,
                chunk_length: 3,
                dp_strategy: vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy,
            },
            &[
                Token::StructVariant {
                    name: "VdafInstance",
                    variant: "Prio3SumVecField64MultiproofHmacSha256Aes128",
                    len: 5,
                },
                Token::Str("proofs"),
                Token::U8(2),
                Token::Str("bits"),
                Token::U64(1),
                Token::Str("length"),
                Token::U64(8),
                Token::Str("chunk_length"),
                Token::U64(3),
                Token::Str("dp_strategy"),
                Token::Struct {
                    name: "Prio3SumVec",
                    len: 1,
                },
                Token::Str("dp_strategy"),
                Token::Str("NoDifferentialPrivacy"),
                Token::StructEnd,
                Token::StructVariantEnd,
            ],
        );
        assert_tokens(
            &VdafInstance::Prio3Histogram {
                length: 6,
                chunk_length: 2,
                dp_strategy: vdaf_dp_strategies::Prio3Histogram::NoDifferentialPrivacy,
            },
            &[
                Token::StructVariant {
                    name: "VdafInstance",
                    variant: "Prio3Histogram",
                    len: 3,
                },
                Token::Str("length"),
                Token::U64(6),
                Token::Str("chunk_length"),
                Token::U64(2),
                Token::Str("dp_strategy"),
                Token::Struct {
                    name: "Prio3Histogram",
                    len: 1,
                },
                Token::Str("dp_strategy"),
                Token::Str("NoDifferentialPrivacy"),
                Token::StructEnd,
                Token::StructVariantEnd,
            ],
        );
        assert_tokens(
            &VdafInstance::Fake { rounds: 17 },
            &[
                Token::StructVariant {
                    name: "VdafInstance",
                    variant: "Fake",
                    len: 1,
                },
                Token::Str("rounds"),
                Token::U32(17),
                Token::StructVariantEnd,
            ],
        );
        assert_tokens(
            &VdafInstance::FakeFailsPrepInit,
            &[Token::UnitVariant {
                name: "VdafInstance",
                variant: "FakeFailsPrepInit",
            }],
        );
        assert_tokens(
            &VdafInstance::FakeFailsPrepStep,
            &[Token::UnitVariant {
                name: "VdafInstance",
                variant: "FakeFailsPrepStep",
            }],
        );
    }

    #[cfg(feature = "fpvec_bounded_l2")]
    #[test]
    fn vdaf_deserialization_backwards_compatibility_fpvec_bounded_l2() {
        assert_eq!(
            serde_yaml::from_str::<VdafInstance>(
                "---
!Prio3FixedPointBoundedL2VecSum
bitsize: BitSize16
dp_strategy:
    dp_strategy: ZCdpDiscreteGaussian
    budget:
        epsilon:
        - - 1
        - - 2
length: 10"
            )
            .unwrap(),
            VdafInstance::Prio3FixedPointBoundedL2VecSum {
                bitsize: Prio3FixedPointBoundedL2VecSumBitSize::BitSize16,
                dp_strategy:
                    vdaf_dp_strategies::Prio3FixedPointBoundedL2VecSum::ZCdpDiscreteGaussian(
                        ZCdpDiscreteGaussian::from_budget(ZCdpBudget::new(
                            Rational::from_unsigned(1u128, 2u128).unwrap(),
                        )),
                    ),
                length: 10,
            }
        );
    }

    #[test]
    fn vdaf_deserialization_backwards_compatibility() {
        assert_matches!(
            serde_yaml::from_str(
                "---
!Prio3Sum
max_measurement: 4096"
            ),
            Ok(VdafInstance::Prio3Sum {
                max_measurement: 4096
            })
        );
        assert_matches!(
            serde_yaml::from_str(
                "---
!Prio3Histogram
length: 4
chunk_length: 2"
            ),
            Ok(VdafInstance::Prio3Histogram {
                length: 4,
                chunk_length: 2,
                dp_strategy: vdaf_dp_strategies::Prio3Histogram::NoDifferentialPrivacy,
            })
        );
        assert_eq!(
            serde_yaml::from_str::<VdafInstance>(
                "---
!Prio3SumVec
bits: 2
length: 2
chunk_length: 2
dp_strategy:
    dp_strategy: PureDpDiscreteLaplace
    budget:
        epsilon: [[1], [1]]"
            )
            .unwrap(),
            VdafInstance::Prio3SumVec {
                bits: 2,
                length: 2,
                chunk_length: 2,
                dp_strategy: vdaf_dp_strategies::Prio3SumVec::PureDpDiscreteLaplace(
                    DiscreteLaplaceDpStrategy::from_budget(
                        PureDpBudget::new(Rational::from_unsigned(1u128, 1u128).unwrap()).unwrap()
                    ),
                ),
            }
        );
    }
}
