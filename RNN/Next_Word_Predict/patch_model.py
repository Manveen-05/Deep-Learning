import h5py
import json

def remove_quantization_config(filepath):
    try:
        with h5py.File(filepath, 'a') as f:
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                config = json.loads(config_str)

                def strip_quantization(obj):
                    """Recursively remove quantization_config from all layers."""
                    if isinstance(obj, dict):
                        obj.pop('quantization_config', None)
                        for v in obj.values():
                            strip_quantization(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            strip_quantization(item)

                strip_quantization(config)

                new_config_str = json.dumps(config).encode('utf-8')
                f.attrs['model_config'] = new_config_str
                print("Successfully patched model_config - removed all quantization_config entries.")
            else:
                print("No model_config found in the h5 file.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    remove_quantization_config('lstm_model.h5')
