# -*- coding: utf-8 -*-
"""
    netstat_script
    ~~~~~~~

    Description.

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2020-08-20
    :python version: 3.5
"""
import os
import re
from collections import OrderedDict
from datetime import datetime

if __name__ == '__main__':
    output_path = '/root/netstat_result.tsv'
    detail_output_path = '/root/netstat_result_detail.tsv'
    cmd = 'netstat -tuWanp | grep 7443'
    # output_path = '/home/zhaoyouqing/Desktop/netstat_result.tsv'
    # detail_output_path = '/home/zhaoyouqing/Desktop/netstat_result_detail.tsv'
    # cmd = 'netstat -tuWanp'

    tcp_statuses = [
        'SYN_SENT', 'LISTEN', 'SYN_RECV', 'ESTABLISHED', 'CLOSING',
        'FIN_WAIT1', 'CLOSE_WAIT', 'FIN_WAIT2', 'LAST_ACK', 'TIME_WAIT',
        'CLOSED'
    ]

    tcp_status_set = set(tcp_statuses)
    tcp_status_count_dict = OrderedDict()
    for _ in tcp_statuses:
        tcp_status_count_dict[_] = 0

    records = list()
    for _ in os.popen(cmd):
        _ = _.strip()
        values = re.split(r'\s+', _)

        for value in values:
            if value not in tcp_status_set:
                continue

            tcp_status_count_dict[value] += 1
            records.append(values)
            break

    now_time = datetime.now()
    if os.path.exists(output_path):
        with open(output_path, 'a+') as f:
            total = 0
            status_count_list = list()
            for _ in tcp_status_count_dict.values():
                total += _
                status_count_list.append(str(_))
            status_content = '\t'.join(status_count_list)

            f.write(
                '%s\t%s\t%s\n' % (
                    now_time.strftime('%Y-%m-%d %H:%M:%S'),
                    status_content, total
                )
            )
    else:
        with open(output_path, 'w') as f:

            f.write('时间\t%s\t总数\n' % '\t'.join(tcp_statuses))

            total = 0
            status_count_list = list()
            for _ in tcp_status_count_dict.values():
                total += _
                status_count_list.append(str(_))
            status_content = '\t'.join(status_count_list)

            f.write(
                '%s\t%s\t%s\n' % (
                    now_time.strftime('%Y-%m-%d %H:%M:%S'),
                    status_content, total
                )
            )

    if os.path.exists(detail_output_path):
        with open(detail_output_path, 'a+') as f:
            for _ in records:
                content = '\t'.join(_)
                f.write(
                    '%s\t%s\n' % (
                        now_time.strftime('%Y-%m-%d %H:%M:%S'), content
                    )
                )

            f.write('\n')
    else:

        netstat_headers = [
            '时间', 'Proto', 'Recv-Q', 'Send-Q', 'Local Address',
            'Foreign Address', 'State', 'PID/Program', 'name'
        ]

        with open(detail_output_path, 'w') as f:

            f.write('%s\n' % '\t'.join(netstat_headers))
            for _ in records:
                content = '\t'.join(_)
                f.write(
                    '%s\t%s\n' % (
                        now_time.strftime('%Y-%m-%d %H:%M:%S'), content
                    )
                )

            f.write('\n')
