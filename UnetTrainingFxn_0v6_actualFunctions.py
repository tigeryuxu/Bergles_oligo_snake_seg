# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:35:56 2019

@author: darya

GENERAL IMAGE TRANSFORMATION FUNCTIONS FOR CNN TRAINING
version 0.6, now *F*L*A*W*L*E*S*S* and sleeker than ever

- as of now contains functions for the following image operations:
    - rescaling
    - gaussian blur
    - noise
    - contrast 
    - stretch in X, Y dimensions
    
- each function generally has a kind, amount, and random argument
    - kind defines the subtype of operation, ie what kind of noise or whether to 
        stretch x vs y dimension
    - amount defines the magnitude, usually btwn 0 and 1 unless its about rescaling
        in which case <1 increases image
    - if random is set to true, it randomly chooses the amount to be between 0 
        and whatever user defined as amount
       
- to-do's
    - make random rescaling and stretching bidirectional - now, it will always
        pick an amount value btwn 0 and amount, so it always skews negative;
        should have it randomly decide whether to increase OR decrease these
        by the given amount
    - make other transformation function (rotation, reflection, shearing)
    - make default values for args (kind=1,amount=1,random=FALSE)

"""


import numpy as np
import scipy # necessary? at this point im just scared to touch anything so it will stay

# rescale image
    # im = image
    # amount = factor to rescale image by (ie 2 means double image size)
    # returns rescaled image
# note - did wierd multiplying-then-dividing-by-10 thing so that 
    # scale2 could be an integer for randint, then go back to being <1
def changeScale(im,amount,random):
    from skimage.transform import rescale
    from random import randint
    from math import floor
    if random == False:
        imScaled = rescale(im,amount,anti_aliasing=True,multichannel=True)
    else: 
        imScaled = rescale(im,amount/randint(1,floor(amount*10))/10, anti_aliasing=True,multichannel=True)
    return imScaled


# gaussian blur
    # im = image
    # amount = proportional to sigma of gaussian; generally btwn 0 and 1
    # random = if true, will apply some blur btwn 0 and amount
    # returns blurred image
# note - gaussian function has other optional parameters including wrap
def changeBlur(im,amount,random):
    from skimage.filters import gaussian
    from random import randint
    if random == False:
        imBlurred = gaussian(im,sigma=amount*10,multichannel=True)
    else:
        imBlurred = gaussian(im,sigma=randint(0,(amount*10)),multichannel=True)
    return imBlurred


# gaussian noise
    # im = image
    # kind: 1=gaussian, 2=speckle, 3=salt and pepper
    # amount = variation of noise (for gaussian and speckle) or amount of 
        # pixels to replace with noise (for salt and pepper); 
    # random = if true, applies noise btwn 0 and amount
    # returns image with added noise
# note - seems like speckle in general is best at emulating immunofluorescence noise
def changeNoise(im,kind,amount,random):
    from skimage.util import random_noise
    from random import randint# was WAY too strong so in interest of keeping things consistent
    if random == True:
        amount = randint(0,amount*10)/10
    if kind == 1:
        amount = amount*0.1     # was too strong otherwise
        imNoised = random_noise(im,mode="gaussian",var=amount)
    if kind == 2:
        imNoised = random_noise(im,mode="speckle",var=amount)
    if kind == 3:
        amount = amount*0.1
        imNoised = random_noise(im,mode="s&p",amount=amount)
    return imNoised


# contrast 
    # kind: 1 = linear equalize, 2 = adaptive equalize, 3 = linear defined
    # amount does nothing for linear equalization, sets the clip limit for CLAHE,
        # and sets the magnitude of floor raising / ceiling lowering for linear defined
    # random = if true, randomly sets amount to be btwn 0 and amount for CLAHE; for
        # linear defined, adds a random number between 0 and half the image range * amount,
        # and subtracts different random number from ceiling
# should rewrite random to increase OR decrease floor and ceiling, as was done with stretch below
# should also allow channel(s) to be specified; does all flattened channels now
def changeContrast(im,kind,amount,random):
    import skimage.exposure
    from random import randint
    from math import floor
    if kind == 1:
        imContrast = skimage.exposure.equalize_hist(im)
    if kind == 2:
        if random == True:
            amount = randint(0,amount*10)/10
        imContrast = skimage.exposure.equalize_adapthist(im,clip_limit=amount)
    if kind == 3:
        imMin = im.min(axis=0).min()
        imMax = im.max(axis=0).max()
        imRange = imMax-imMin  # this takes a percentage of the total image range to add to floor or subtract from ceiling
        if random == True:
            newMin = imMin + randint(0,floor(amount*imRange/2))
            newMax = imMax - randint(0,floor(amount*imRange/2))
        else:
            newMin = imMin + amount*imRange/2
            newMax = imMax - amount*imRange/2
        imContrast = skimage.exposure.rescale_intensity(im,in_range='image',out_range=(newMin.astype(int),newMax.astype(int)))
    return imContrast


# stretch image 
    # kind = (1 = X dimension only, 2 = Y only, 3 = both)
    # amount is factor to stretch by, ie 0.5 means that dimension will be halved
    # random = if true, sets random value btwn 0 and amount 
        # if both axes are stretched, will find a random value for each, so NOT
        # the same as random rescaling where its all proportional 
        # however, non-random X and Y stretching is identical to scaling
# may need to add in something to reshape image to original size after...
def changeStretch(im,kind,amount,random):
    from skimage.transform import resize
    from random import randint
    from math import ceil
    if kind == 1:
        if random == True:
            amount = randint(0,amount*10)/10
        imStretch = resize(im,(im.shape[0]*amount,im.shape[1]))    
    if kind == 2:  
        if random == True:
            amount = randint(0,amount*10)/10
        imStretch = resize(im,(im.shape[1],im.shape[1]*amount))
    if kind == 3: 
        if random == True:
            changeX = ceil(im.shape[0]*randint(1,amount*10)/10)
            changeY = ceil(im.shape[1]*randint(1,amount*10)/10)
        else:
            changeX = ceil(im.shape[0]*amount)
            changeY = ceil(im.shape[1]*amount)
        imStretch = resize(im,(changeX,changeY))
    return imStretch


# reflect image
    # kind = (1 = reflect X axis, 2 = reflect Y, 3 = reflect both)
    # random = if true, ignores kind and chooses a random reflection
def changeFlip(im,kind,random):
    from random import randint
    from numpy import flip
    if random == True:
        kind = randint(1,3)
    if kind ==1:
        imFlip = flip(im,axis=0)
    if kind == 2:
        imFlip = flip(im,axis=1)
    if kind == 3:
        imFlip = flip(im,axis=0)
        imFlip = flip(im,axis=1)
    return imFlip


# rotate image
    # kind = (1 = 90 degrees (maintain size), 2 = any degree w/out size compensation, 
        # 3 = any degree with resizing)
    # amount = radians to rotate (ie 1 means 180deg, 1.5 means 270 aka -90)
        # if kind=1, amount will be rounded to nearest multiple of 0.5
    # random = chooses random amount
    # not sure if this is useful so leaving it for now
