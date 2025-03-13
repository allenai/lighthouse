from enum import StrEnum

from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client.context_managers import Timer

request_timer = Histogram('lighthouse_request_timer',
                          'Timers for d2c lookup requests',
                          ['operation'])


class TimerOperations(StrEnum):
    TotalBatchLookup = 'total_batch_lookup'
    TotalSingleLookup = 'total_single_lookup'
    BatchLandTileLookups = 'batch_land_tile_lookups'
    LookupNearestCoast = 'lookup_nearest_coast'
    GetBallTree = 'get_ball_tree'
    LoadH5File = 'load_h5_file'


def time_operation(operation: TimerOperations) -> Timer:
    return request_timer.labels(operation=operation.value).time()
