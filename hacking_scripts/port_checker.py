import socket 

# definition of the socket
socket_definition = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# defining the server
server = 'qualitestgroup.com'sentdex

# function takes  a port number and returns true if it exists and false if 
# it does not
def portscan(port):
	try:
		socket_definition.connect((server, port))
		return True
	except:
		return False

# loop tries ports from a range by calling 'portscan'
for x in range(1, 26):
	if portscan(x):
		print('port', x, 'is open')
	else:
		print('ports', x, 'is closed')
