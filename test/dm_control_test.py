import os
from dm_control import mujoco
_JOINTS = ['cabin', 'boom', 'arm', 'bucket']
xml_path = os.path.join(os.path.dirname(__file__), "../assets/excavator.xml")
physics = mujoco.Physics.from_xml_path(xml_path)
xpos = physics.named.data.site_xpos['dig_point']
print(xpos)