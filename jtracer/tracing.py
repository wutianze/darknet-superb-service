import logging
from jaeger_client import Config


def init_tracer(service):
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    config = Config(
        config={ # usually read from some yaml config
            'sampler': {
                'type': 'const',
                'param': 1,
            },
            'logging': True,
            'reporter_batch_size': 1,
            'local_agent': {
                'reporting_host': "10.16.0.180",
                'reporting_port': 5775,
            }
        },
        service_name=service,
    )

    return config.initialize_tracer()
