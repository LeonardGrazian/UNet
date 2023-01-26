
from ray import tune
# from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from main import create_model


def main():
    # search_space = {
    #     'encoder_channels': tune.grid_search([
    #         [32, 64, 128, 256],
    #         [64, 128, 256, 512],
    #         [128, 256, 512, 1024]
    #     ]),
    #     'bottleneck_logits': tune.grid_search([512, 1024, 2048]),
    #     'decoder_channels': tune.grid_search([
    #         [32, 64, 128, 256],
    #         [64, 128, 256, 512],
    #         [128, 256, 512, 1024]
    #     ]),
    #     'lr': tune.loguniform(1e-5, 1e-1),
    #     'batch_size': tune.choice([16, 32, 64])
    # }
    search_space = {
        'encoder_channels': [64, 128, 256, 512],
        'bottleneck_logits': 1024,
        'decoder_channels': [64, 128, 256, 512],
        'lr': tune.loguniform(1e-5, 1e-1),
        'batch_size': tune.choice([16, 32, 64])
    }

    tuner = tune.Tuner(
        create_model,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric='loss',
            mode='min',
            num_samples=10
        )
    )
    results = tuner.fit()
    print(results.get_best_result(metric='loss', mode='min').config)


if __name__ == '__main__':
    main()
