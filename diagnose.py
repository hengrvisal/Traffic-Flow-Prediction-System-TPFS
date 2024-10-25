import json
import h5py
from keras.models import model_from_json

with h5py.File('model/sites_models/saes_970.h5', 'r') as f:
    # Manually read the model config
    model_config = f.attrs.get('model_config')
    if isinstance(model_config, bytes):
        model_config = model_config.decode('utf-8')
    model = model_from_json(json.loads(model_config))

    # Load the model weights
    model.load_weights(f)
