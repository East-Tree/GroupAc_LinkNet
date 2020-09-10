import pickle
import math
import torch


def maxSlidingWindow(nums, k):
    if k == 0: return []
    arr = [0] * (len(nums) - k + 1)
    arr[0] = max(nums[0:k])
    for i in range(1, len(nums) - k + 1):
        if nums[i + k - 1] >= arr[i - 1]:
            arr[i] = nums[i + k - 1]
        else:
            if nums[i - 1] == arr[i - 1]:
                arr[i] = max(nums[i:i + 3])
            else:
                arr[i] = arr[i - 1]
    return arr

maxSlidingWindow([5,3,4],1)
