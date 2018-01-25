import glob
import json
from datetime import datetime

def _is_valid(battle, start, end):
    return __is_automated(battle) and __is_ranked_gachi(battle) and __is_within_pediod(battle, start, end) and not __is_disconnected(battle)

def __is_automated(battle):
    if battle['automated']:
        return True
    else:
        try:
            name = battle['agent']['name']
            if name is None:
                return False
        except KeyError as e:
            return False

        return 'splatnet2statink' in name

def __is_ranked_gachi(battle):
    return battle['lobby'] is not None and battle['lobby'].get('key') == 'standard' \
           and battle['mode'] is not None and battle['mode'].get('key') == 'gachi'

def __is_within_pediod(battle, start, end):
    return battle['start_at'] is not None and start <= datetime.fromtimestamp(battle['start_at']['time']) <= end

def __is_disconnected(battle):
    if battle['players'] is None:
        return True
    elif len(battle['players']) < 8:
        return True
    else:
        for p in battle['players']:
            if p['point'] == 0:
                return True
    return False

def _process_battle_data(battle):
    try:
        my_weapons = [p['weapon']['reskin_of'] or p['weapon']['key'] for p in battle['players'] if p['team'] == 'my']
        his_weapons = [p['weapon']['reskin_of'] or p['weapon']['key'] for p in battle['players'] if p['team'] == 'his']
        stage = battle['map']['key']
        rule = battle['rule']['key']
        rank = battle['rank']['zone']['key']
        if battle['rank']['key'] == 's+':
            rank = 's+'
        gachi_power = battle['estimate_gachi_power']
        result = battle['result']
        return my_weapons + his_weapons + [stage, rule, rank, gachi_power, result]
    except (TypeError, KeyError) as e:
        return None

def extract_valid_battle_data(battles, start, end):
    return [_process_battle_data(battle) for battle in battles if _is_valid(battle, start, end)]

def extract_from_json_files(json_dir, start, end):
    json_files = glob.glob(json_dir+'/stat*.json')
    print(json_files)
    dats = []
    for ajson in json_files:
        with open(ajson) as f:
            battles = json.load(f)
            dats.extend(extract_valid_battle_data(battles, start, end))
    return dats

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                        help='stat.ink形式のjsonが存在するディレクトリを指定します',
                        type=str)
    parser.add_argument('--start', '-s',
                         help='集計対象に含める期間の開始時刻を%YYYY-%mm-%dd-%HHで指定します',
                         type=str)
    parser.add_argument('--end', '-e',
                         help='集計対象に含める期間の終了時刻を%YYYY-%mm-%dd-%HHで指定します',
                         type=str)
    args = parser.parse_args()

    start_at = datetime.strptime(args.start, "%Y-%m-%d-%H")
    end_at = datetime.strptime(args.end, "%Y-%m-%d-%H")
    if start_at > end_at:
        raise ValueError('must be start_time <= end_time')
    dats = extract_from_json_files(args.data_dir, start_at, end_at)
    print(len(dats))
    print(sum(x is None for x in dats))