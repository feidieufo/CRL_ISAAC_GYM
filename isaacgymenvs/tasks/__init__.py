# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from tasks.allegro_hand import AllegroHand
from tasks.ant import Ant
from tasks.anymal import Anymal
from tasks.anymal_terrain import AnymalTerrain
from tasks.ball_balance import BallBalance
from tasks.cartpole import Cartpole 
from tasks.franka_cabinet import FrankaCabinet
from tasks.humanoid import Humanoid
from tasks.humanoid_amp import HumanoidAMP
from tasks.ingenuity import Ingenuity
from tasks.quadcopter import Quadcopter
from tasks.shadow_hand import ShadowHand
from tasks.trifinger import Trifinger
from tasks.humanoid_cost import HumanoidCost
from tasks.ant_cost import AntCost
from tasks.ant_speed_cost import AntSpeedCost
from tasks.humanoid_speed_cost import HumanoidSpeedCost
from tasks.franka_cabinet_cost import FrankaCabinetCost
from tasks.ant_2_cost import Ant2Cost
from tasks.humanoid_2_cost import Humanoid2Cost
from tasks.shadow_hand_cost import ShadowHandCost
from tasks.anymal_2_cost import Anymal2Cost
from tasks.anymal_terrain_2_cost import AnymalTerrain2Cost

# Mappings from strings to environments
isaacgym_task_map = {
    "AllegroHand": AllegroHand,
    "Ant": Ant,
    "Anymal": Anymal,
    "AnymalTerrain": AnymalTerrain,
    "BallBalance": BallBalance,
    "Cartpole": Cartpole,
    "FrankaCabinet": FrankaCabinet,
    "Humanoid": Humanoid,
    "HumanoidAMP": HumanoidAMP,
    "Ingenuity": Ingenuity,
    "Quadcopter": Quadcopter,
    "ShadowHand": ShadowHand,
    "Trifinger": Trifinger,
    "HumanoidCost": HumanoidCost,
    "AntCost": AntCost,
    "AntSpeedCost": AntSpeedCost,
    "HumanoidSpeedCost": HumanoidSpeedCost,
    "FrankaCabinetCost": FrankaCabinetCost,
    "Ant2Cost": Ant2Cost,
    "Humanoid2Cost": Humanoid2Cost,
    "ShadowHandCost": ShadowHandCost,
    "Anymal2Cost": Anymal2Cost,
    "AnymalTerrain2Cost": AnymalTerrain2Cost,
}