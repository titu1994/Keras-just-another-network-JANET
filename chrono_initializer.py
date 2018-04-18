from keras.utils.generic_utils import get_custom_objects
from keras import initializers
from keras import backend as K


class ChronoInitializer(initializers.RandomUniform):

    def __init__(self, max_timesteps, seed=None):
        super(ChronoInitializer, self).__init__(1., max_timesteps - 1, seed)
        self.max_timesteps = max_timesteps

    def __call__(self, shape, dtype=None):
        values = super(ChronoInitializer, self).__call__(shape, dtype=dtype)
        return K.log(values)

    def get_config(self):
        config = {
            'max_timesteps': self.max_timesteps
        }
        base_config = super(ChronoInitializer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'ChronoInitializer': ChronoInitializer,
                             'chrono_initializer': ChronoInitializer})
