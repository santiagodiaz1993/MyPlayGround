import socket 
# Define the socket. Socket is the "opening of a session" for 
# filing requests to a server. 'AT_INET' = connection type, and
# 'socket.SOCK_STREAM' means TCP connection type (the most popular).
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the server and the port
server = 'qualitestgroup.com'
port = 80

# get the IP address from the domain name
server_ip = socket.gethostbyname(server)
print(server_ip)

# write the request in structured format.
request = 'GET / HTTP/1.1\nHost: '+server+'\n\n'

# connect to the server figen the somain name and the port
serversocket.connect((server, port))

# send the request given the server plus the right format 
# and encoded
serversocket.send(request.encode())

 # record the result with '.recv'
result = serversocket.recv(4096) 

print(result)


# while (len(result) > 0):
# 	print(result)
# 	result = serversocket.recv(4096)