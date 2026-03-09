from . import gaussian_diffusion as gd
from . import gaussian_diffusion_seq_interpolation_sampling as gd_seq_interpolation_sampling
from . import gaussian_diffusion_seq_three_row as gd_seq_three_row
from .respace import SpacedDiffusion, space_timesteps
from .respace_seq_interpolation_sampling import SpacedDiffusion_seq_interpolation_sampling, space_timesteps_seq_interpolation_sampling
from .respace_seq_three_row import SpacedDiffusion_seq_three_row, space_timesteps_seq_three_row

def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )

def create_diffusion_seq_three_row(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = gd_seq_three_row.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd_seq_three_row.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd_seq_three_row.LossType.RESCALED_MSE
    else:
        loss_type = gd_seq_three_row.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion_seq_three_row(
        use_timesteps=space_timesteps_seq_three_row(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd_seq_three_row.ModelMeanType.EPSILON if not predict_xstart else gd_seq_three_row.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_seq_three_row.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd_seq_three_row.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_seq_three_row.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )

def create_diffusion_seq_interpolation_sampling(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = gd_seq_interpolation_sampling.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd_seq_interpolation_sampling.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd_seq_interpolation_sampling.LossType.RESCALED_MSE
    else:
        loss_type = gd_seq_interpolation_sampling.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion_seq_interpolation_sampling(
        use_timesteps=space_timesteps_seq_interpolation_sampling(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd_seq_interpolation_sampling.ModelMeanType.EPSILON if not predict_xstart else gd_seq_interpolation_sampling.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_seq_interpolation_sampling.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd_seq_interpolation_sampling.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_seq_interpolation_sampling.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )    