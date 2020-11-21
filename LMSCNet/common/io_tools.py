import hashlib
import os


def get_md5(filename):
  '''

  '''
  hash_obj = hashlib.md5()
  with open(filename, 'rb') as f:
      hash_obj.update(f.read())
  return hash_obj.hexdigest()


def dict_to(_dict, device, dtype):
  '''

  '''
  for key, value in _dict.items():
    if type(_dict[key]) is dict:
      _dict[key] = dict_to(_dict[key], device, dtype)
    else:
      _dict[key] = _dict[key].to(device=device, dtype=dtype)

  return _dict


def _remove_recursively(folder_path):
  '''
  Remove directory recursively
  '''
  if os.path.isdir(folder_path):
    filelist = [f for f in os.listdir(folder_path)]
    for f in filelist:
      os.remove(os.path.join(folder_path, f))
  return


def _create_directory(directory):
  '''
  Create directory if doesn't exists
  '''
  if not os.path.exists(directory):
    os.makedirs(directory)
  return