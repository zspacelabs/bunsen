//! A stage: residual units in sequence.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    config::Config,
    module::Module,
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    blocks::images::models::conv_net::{
        ResidualUnit,
        ResidualUnitConfig,
        ResidualUnitMeta,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        ResultContext,
        Rule,
    },
    ops::drop::DropBlockOptions,
};

/// [`ConvStageConfig`]'s and [`ConvStage`]'s shared view, beside
/// [`ResidualUnitMeta`]'s: how many units.
pub trait ConvStageMeta: ResidualUnitMeta {
    /// The number of units.
    fn len(&self) -> usize;

    /// Whether the stage has no units.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// [`ConvStage`]'s config: its units, each one resolved.
///
/// Implements [`ConvStageMeta`].
#[derive(Config, Debug)]
pub struct ConvStageConfig {
    /// The units, in order; each one's output is the next one's input.
    pub units: Vec<ResidualUnitConfig>,
}

impl From<Vec<ResidualUnitConfig>> for ConvStageConfig {
    fn from(units: Vec<ResidualUnitConfig>) -> Self {
        Self { units }
    }
}

impl ResidualUnitMeta for ConvStageConfig {
    fn in_channels(&self) -> usize {
        self.units[0].in_channels()
    }

    fn out_channels(&self) -> usize {
        self.units[self.units.len() - 1].out_channels()
    }

    fn stride(&self) -> usize {
        self.units.iter().map(|u| u.stride()).product()
    }
}

impl ConvStageMeta for ConvStageConfig {
    fn len(&self) -> usize {
        self.units.len()
    }
}

impl ConvStageConfig {
    /// Checks the stage: units that are not empty, each valid, and chaining.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause; a unit's own error under a frame naming it.
    pub fn try_validate(&self) -> BunsenResult<()> {
        if self.units.is_empty() {
            return Err(ConstraintError::zero_or_empty("ConvStageConfig", "units").into());
        }
        for (idx, unit) in self.units.iter().enumerate() {
            unit.try_validate()
                .with_context(|| format!("unit {idx} (units[{idx}])"))?;
            if idx > 0 {
                let prev = &self.units[idx - 1];
                if prev.out_channels() != unit.in_channels() {
                    return Err(ConstraintError::new(
                        "ConvStageConfig",
                        "units",
                        Rule::Chain {
                            index: idx,
                            out: format!("{} channels", prev.out_channels()),
                            r#in: format!("{} channels", unit.in_channels()),
                        },
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    /// Applies `f` to each unit, with its index in the stage.
    pub fn map_units<F>(
        self,
        f: &mut F,
    ) -> Self
    where
        F: FnMut(usize, ResidualUnitConfig) -> ResidualUnitConfig,
    {
        Self {
            units: self
                .units
                .into_iter()
                .enumerate()
                .map(|(idx, unit)| f(idx, unit))
                .collect(),
        }
    }

    /// Sets every unit's drop block, as [`ConvStage::with_drop_block`] does
    /// to a built stage.
    pub fn with_drop_block(
        self,
        options: Option<DropBlockOptions>,
    ) -> Self {
        self.map_units(&mut |_, unit| unit.with_drop_block(options.clone()))
    }
}

impl ModuleInit<ConvStage> for ConvStageConfig {
    /// Builds the stage.
    ///
    /// # Errors
    ///
    /// As [`try_validate`](ConvStageConfig::try_validate).
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConvStage> {
        self.try_validate()?;
        Ok(ConvStage {
            units: self
                .units
                .iter()
                .map(|u| u.try_init(device))
                .collect::<BunsenResult<_>>()?,
        })
    }
}

/// A conv net's stage: [`ResidualUnit`]s in sequence; the first changes the
/// width or the stride. Built by [`ConvStageConfig`].
///
/// Implements [`ConvStageMeta`].
#[derive(Module, Debug)]
pub struct ConvStage {
    /// The units, in order.
    pub units: Vec<ResidualUnit>,
}

impl ResidualUnitMeta for ConvStage {
    fn in_channels(&self) -> usize {
        self.units[0].in_channels()
    }

    fn out_channels(&self) -> usize {
        self.units[self.units.len() - 1].out_channels()
    }

    fn stride(&self) -> usize {
        self.units.iter().map(|u| u.stride()).product()
    }
}

impl ConvStageMeta for ConvStage {
    fn len(&self) -> usize {
        self.units.len()
    }
}

impl ConvStage {
    /// Applies the units in order.
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, in_channels, in_height, in_width]`.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_height, out_width]`, each side
    /// `ceil(in / stride)`, as torch's.
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        self.units.iter().fold(input, |x, unit| unit.forward(x))
    }

    /// Applies `f` to each unit, with its index in the stage.
    pub fn map_units<F>(
        self,
        f: &mut F,
    ) -> Self
    where
        F: FnMut(usize, ResidualUnit) -> ResidualUnit,
    {
        Self {
            units: self
                .units
                .into_iter()
                .enumerate()
                .map(|(idx, unit)| f(idx, unit))
                .collect(),
        }
    }

    /// Sets every unit's drop block; none clears them.
    pub fn with_drop_block(
        self,
        options: Option<DropBlockOptions>,
    ) -> Self {
        self.map_units(&mut |_, unit| unit.with_drop_block(options.clone()))
    }
}
