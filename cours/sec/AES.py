from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

data = b"understanding the AES algorithm"
key = get_random_bytes(16)
iv = get_random_bytes(16)


cipher = AES.new(key, AES.MODE_CBC, iv)
ciphertext = cipher.encrypt(pad(data, AES.block_size))

decipher = AES.new(key, AES.MODE_CBC, iv)
plaintext = unpad(decipher.decrypt(ciphertext), AES.block_size)

print("Message original :", data)
print("Message chiffré :", ciphertext)
print("Message déchiffré :", plaintext)