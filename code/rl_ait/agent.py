import os
try:
    from lux_ait.game import Game
    from lux_ait.game_map import Cell, RESOURCE_TYPES, Position
    from lux_ait.game_objects import Unit
    from lux_ait.constants import Constants
    from lux_ait.game_constants import GAME_CONSTANTS
    from lux_ait import annotate
except:
    from rl_ait.lux_ait.game import Game
    from rl_ait.lux_ait.game_map import Cell, RESOURCE_TYPES, Position
    from rl_ait.lux_ait.game_objects import Unit
    from rl_ait.lux_ait.constants import Constants
    from rl_ait.lux_ait.game_constants import GAME_CONSTANTS
    from rl_ait.lux_ait import annotate
import math, sys
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque


from pathlib import Path
p = Path('/kaggle_simulations/agent/')
if p.exists():
    sys.path.append(str(p))
else:
    p = Path('__file__').resolve().parent


game_state = None

model = None

def get_inputs(game_state):
    # Teh shape of the map
    w,h = game_state.map.width, game_state.map.height
    # The map of ressources
    M = [ [0  if game_state.map.map[j][i].resource==None else game_state.map.map[j][i].resource.amount for i in range(w)]  for j in range(h)]
    
    M = np.array(M).reshape((h,w,1))
    
    # The map of units features
    U_player = [ [[0,0,0,0,0] for i in range(w)]  for j in range(h)]    
    units = game_state.player.units
    for i in units:
        U_player[i.pos.y][i.pos.x] = [i.type,i.cooldown,i.cargo.wood,i.cargo.coal,i.cargo.uranium]
    U_player = np.array(U_player)
    
    U_opponent = [ [[0,0,0,0,0] for i in range(w)]  for j in range(h)]
    units = game_state.opponent.units
    for i in units:
        U_opponent[i.pos.y][i.pos.x] = [i.type,i.cooldown,i.cargo.wood,i.cargo.coal,i.cargo.uranium]

    U_opponent = np.array(U_opponent)
    
    # The map of cities featrues
    e = game_state.player.cities
    C_player = [ [[0,0,0] for i in range(w)]  for j in range(h)]
    for k in e:
        citytiles = e[k].citytiles
        for i in citytiles:
            C_player[i.pos.y][i.pos.x] = [i.cooldown,e[k].fuel,e[k].light_upkeep]
    C_player = np.array(C_player)

    e = game_state.opponent.cities
    C_opponent = [ [[0,0,0] for i in range(w)]  for j in range(h)]
    for k in e:
        citytiles = e[k].citytiles
        for i in citytiles:
            C_opponent[i.pos.y][i.pos.x] = [i.cooldown,e[k].fuel,e[k].light_upkeep]
    C_opponent = np.array(C_opponent)
    
    # stacking all in one array
    E = np.dstack([M,U_opponent,U_player,C_opponent,C_player])
    return E


def get_model(s):
    input_shape = (s,s,17)
    inputs = keras.Input(shape= input_shape,name = 'the_game_map')
    f = layers.Flatten()(inputs)   
    h,w,_ = get_inputs(game_state).shape
    print(h,w)
#     output = layers.Dense(w*h*8,activation = "sigmoid")(f)
    
    f = layers.Dense(w*h,activation = "sigmoid")(f)
    f = layers.Reshape((h,w,-1))(f)
    units = layers.Dense(6,activation = "softmax",name = "Units_actions")(f)
    
    cities = layers.Dense(2,activation = "sigmoid",name = "Cities_actions")(f)
    
    output = layers.Concatenate()([units,cities])
    model = keras.Model(inputs = inputs, outputs = output)
    return model



def get_prediction_actions(y,player):
    # move
    option = np.argmax(y,axis = 2) 
    # c s n w e build_city & research & buid_worker  
    actions = []
    for i in player.units:
#         print(option.shape,i.pos.y,i.pos.x)
        d = "csnwe#############"[option[i.pos.y,i.pos.x]]
        if option[i.pos.y,i.pos.x]<5:actions.append(i.move(d))
        elif option[i.pos.y,i.pos.x]==5 and i.can_build(game_state.map):actions.append(i.build_city())
    
    for city in player.cities.values():
        for city_tile in city.citytiles:
            if option[city_tile.pos.y,city_tile.pos.x]==6:
                action = city_tile.research()
                actions.append(action)
            if option[city_tile.pos.y,city_tile.pos.x]==7:
                action = city_tile.build_worker()
                actions.append(action)
    return actions,option

def agent(observation, configuration):
    global game_state,epsilon,model
    
    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
        print("Creating model..")
        model =get_model(game_state.map.width)
        print("Load model weight..")

        weight_path = str(p/('model_%d.h5'%game_state.map.width))
        if 'rl_ait' in os.listdir():
            weight_path = os.path.join('rl_ait', weight_path)

        try:
            model.load_weights(weight_path,  by_name=True, skip_mismatch=True)
        except Exception as e:
            print('Error in model load')
            print(e)
            print(weight_path)
#         model = tf.keras.models.load_model('model.h5')
        print("Done creating model")
        
        
    else:
        game_state._update(observation["updates"])
    

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    # Get Prediction of actions
    x = get_inputs(game_state)
    y = model.predict(np.asarray([x]))[0]
    actions,_ = get_prediction_actions(y,player)
    return actions
