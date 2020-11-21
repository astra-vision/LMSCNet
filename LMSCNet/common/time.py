import datetime


def get_date_sting():
  '''
  To retrieve time in nice format for string printing and naming
  :return:
  '''
  _now = datetime.datetime.now()
  _date = ('%.2i' % _now.month) + ('%.2i' % _now.day) # ('%.4i' % _now.year) +
  _time = ('%.2i' % _now.hour) + ('%.2i' % _now.minute) + ('%.2i' % _now.second)
  return (_date + '_' + _time)