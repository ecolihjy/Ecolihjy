from operator import truediv

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

while not outlet.have_consumers():
    print('No consumers connected yet...')
#
# print('Connected')
# while True:
list = [0, 7]
list2 = [0, 0, 0, 0, 0, 0, 0, 10]
list3 = [0, 0, 0, 0, 19]
list4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6]
list5 = [0, 7]
list6 = [0, 0, 0, 0, 0, 0, 0, 10]
list7 = [0, 0, 0, 0, 19]
list8 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6]
list9 = [0, 7]
list10 = [0, 0, 0, 0, 0, 0, 0, 10]
list11 = [0, 0, 0, 0, 19]
list12 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6]
for i in list:
    time.sleep(0.1)
    outlet.push_sample([i], 1)
time.sleep(2)
for i in list2:
    time.sleep(0.1)
    outlet.push_sample([i], 1)
time.sleep(2)
for i in list3:
    time.sleep(0.1)
    outlet.push_sample([i], 1)
time.sleep(2)
for i in list4:
    time.sleep(0.1)
    outlet.push_sample([i], 1)
time.sleep(2)
for i in list5:
    time.sleep(0.1)
    outlet.push_sample([i], 1)
time.sleep(2)
for i in list6:
    time.sleep(0.1)
    outlet.push_sample([i], 1)
time.sleep(2)
for i in list7:
    time.sleep(0.1)
    outlet.push_sample([i], 1)
time.sleep(2)
for i in list8:
    time.sleep(0.1)
    outlet.push_sample([i], 1)
time.sleep(2)
for i in list9:
    time.sleep(0.1)
    outlet.push_sample([i], 1)
time.sleep(2)
for i in list10:
    time.sleep(0.1)
    outlet.push_sample([i], 1)
time.sleep(2)
for i in list11:
    time.sleep(0.1)
    outlet.push_sample([i], 1)
time.sleep(2)
for i in list12:
    time.sleep(0.1)
    outlet.push_sample([i], 1)
time.sleep(2)
    # print('success,Sent data')
    # time.sleep(2)
# i = 0
# while True:
#     outlet.push_sample([2,1],1)
#     i += 1
#     print(f'Sent data,{i}')
#     time.sleep(5)
