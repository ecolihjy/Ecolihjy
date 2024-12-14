from pylsl import StreamInfo, StreamOutlet
import time

lsl_source_id = 'meta_online_worker'
 

info = StreamInfo(
    name='meta_feedback',
    type='Markers',
    channel_count=1,
    nominal_srate=0,
    channel_format='int32',
    source_id=lsl_source_id)

outlet = StreamOutlet(info)
print('Waiting connection...')

print('Waiting for connection...')


while not outlet.have_consumers():
    print('No consumers connected yet...')
    time.sleep(1)

print('Connected')


outlet.push_sample([True])#向出口推送一个值为 [True] 的样本。