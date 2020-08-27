# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: Aug 2017
"""

from collections import OrderedDict as odict


class FlowNet:
    def __init__(self, SimEnv_instance, part, sub_net=False):
        # depending on primary or sub/secondary net:
        if not sub_net:
            # if primary net:
            self.net_type = 'primary flow net'
            # save parent_pump:
            self.parent_pump = part
        elif sub_net:
            # if sub_net:
            self.net_type = 'secondary flow net'
            # save parent_part:
            self.parent_part = part
        # general stuff for both types:
        # get memoryview to massflow array of parent part by slicing:
        self.dm_parent = SimEnv_instance.parts[part].dm[:]
        # from parent part to child part ordered topology dict which determines
        # the order to solve for the massflows. the key is the part/port to
        # solve,the value contains a tuple of the following setup:
        # (memory view to massflow to solve for,
        #  operation id depicting the calculation method,
        #  memory view(s) to source massflow array cell(s)).
        self.dm_topo = odict()
