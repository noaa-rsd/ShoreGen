import math

# http://bytes.com/topic/python/answers/661631-bytes-file-size-format-function

"""Returns a humanized string for a given amount of bytes"""
def filesizeformat(bytes, precision=2):

    bytes = int(bytes)  

    if bytes is 0:
        return '0bytes'

    log = math.floor(math.log(bytes, 1024))

    return "%.*f%s" % (precision, bytes / math.pow(1024, log), ['bytes', 'kb', 'mb', 'gb', 'tb','pb', 'eb', 'zb', 'yb'] [int(log)] )