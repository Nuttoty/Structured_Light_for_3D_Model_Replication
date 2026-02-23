def get_gray_1d(n):
    if n == 1: return ['0', '1']
    prev = get_gray_1d(n - 1)
    return ['0' + s for s in prev] + ['1' + s for s in prev[::-1]]

if __name__ == "__main__":
    print(get_gray_1d(11))