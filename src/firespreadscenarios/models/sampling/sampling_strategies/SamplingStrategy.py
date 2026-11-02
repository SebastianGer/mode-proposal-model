from abc import ABC, abstractmethod


class SamplingStrategy(ABC):

    @abstractmethod
    def sample(self, diffusion_model, latents, replicated_cond, hparams):
        """
        Sample from the diffusion model.

        Args:
            diffusion_model: The diffusion model to sample from.
            latents: Latent variables to start sampling from.
            replicated_cond: Condition to replicate for sampling.
            hparams: Hyperparameters for the sampling process.

        Returns:
            A dictionary containing the sampling results.
        """
        pass

    def additional_metrics(self, sampling_results, batch_dict, i):
        """
        Compute additional metrics based on the results of the sampling process.

        Returns:
            A dictionary containing named metrics, specific to the sampling strategy.
        """
        return {}

    def aggregate_additional_metrics(self, additional_metrics_agg):
        """
        Aggregates the results of self.additional_metrics in a dictionary.
        """
        return {}
