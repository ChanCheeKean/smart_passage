from utils.cv_helper import gate_xyxy

### info logging ###
def determine_zone(det):
    '''check the zone of the passanger or object'''
    center = (det[0] + det[2]) / 2
    if center < gate_xyxy['left-safety']:
        return 'left'
    elif center > gate_xyxy['right-safety']:
        return 'right'
    else:
        return 'safety'

def store_id(passenger_count, id_paid, id_complete, obj_id, zone, paid_zone='right'):
    anti_flag = 0
    if zone == paid_zone:
        if obj_id not in id_paid:
            id_paid.append(obj_id)

    elif (zone != paid_zone) & (zone != 'safety'):
        if (obj_id in id_paid) and (obj_id not in id_complete):
            passenger_count += 1
            id_complete.append(obj_id)

        elif obj_id not in id_paid:
            anti_flag = 1

    return passenger_count, id_paid, id_complete, anti_flag

def update_info(passenger_count, id_paid, id_complete, dets):
    '''combine the info to be displayed in dashboard'''
    info_dict = {
        'human_count' : 0, 
        'human_left' : 0,
        'human_safety': 0,
        'human_right': 0,
        'object_left' : 0,
        'object_safety': 0,
        'object_right': 0,
        'tailgate_flag': 0,
        'antidir_flag': 0,
    }

    for det in dets:
        obj_cls = det[5].item()
        object_id = det[6].item()
        zone = determine_zone(det)
        
        if obj_cls == 0:
            info_dict['human_count'] += 1
            info_dict[f'human_{zone}'] += 1
            passenger_count, id_paid, id_complete, anti_flag = store_id(passenger_count, id_paid, id_complete, object_id, zone, paid_zone='right')
            info_dict['antidir_flag'] = anti_flag
        else:
            info_dict[f'object_{zone}'] = 1

    return passenger_count, id_paid, id_complete, info_dict