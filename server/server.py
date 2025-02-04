import socket
import pickle
import numpy as np

HOST = "0.0.0.0"
PORT = 12345
BUFFER_SIZE = 4096

def compute_gradients_numpy(received_gradients):
    """ Simulates ML updates by modifying received gradients. """
    processed_gradients = []
    for grad in received_gradients:
        grad_np = np.array(grad, dtype=np.float32)
        grad_np *= 0.9  # Simulates gradient update
        processed_gradients.append(grad_np)
    return processed_gradients

def start_server():
    """Starts the Raspberry Pi server and keeps it running indefinitely."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"🚀 Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = server_socket.accept()
            print(f"🔗 Connected to client at {addr}")

            try:
                while True:
                    # Step 1: Receive data size
                    size_data = conn.recv(8)
                    if not size_data:
                        print("❌ Connection lost. Waiting for new client...")
                        break  

                    data_size = int.from_bytes(size_data, "big")
                    print(f"📩 Expecting {data_size} bytes from client...")

                    # Step 2: Receive the full data in chunks
                    data = b""
                    received_bytes = 0
                    while len(data) < data_size:
                        chunk = conn.recv(min(BUFFER_SIZE, data_size - len(data)))
                        if not chunk:
                            print("⚠️ Connection lost while receiving data. Retrying...")
                            break
                        data += chunk
                        received_bytes += len(chunk)
                        print(f"✅ Received {received_bytes}/{data_size} bytes...")  # Debugging

                    if len(data) < data_size:
                        print(f"⚠️ Error: Incomplete data received ({len(data)} bytes instead of {data_size}).")
                        break

                    # Step 3: Deserialize gradients
                    gradients = pickle.loads(data)
                    print(f"📊 Received {len(gradients)} gradients.")

                    # Step 4: Process gradients
                    processed_gradients = compute_gradients_numpy(gradients)

                    # Step 5: Serialize processed gradients
                    serialized_response = pickle.dumps(processed_gradients)
                    response_size = len(serialized_response)

                    # Step 6: Send the size of the processed gradients first
                    conn.sendall(response_size.to_bytes(8, "big"))

                    # Step 7: Send processed gradients in chunks
                    sent_bytes = 0
                    for i in range(0, response_size, BUFFER_SIZE):
                        conn.sendall(serialized_response[i:i+BUFFER_SIZE])
                        sent_bytes += min(BUFFER_SIZE, response_size - sent_bytes)
                        print(f"✅ Sent {sent_bytes}/{response_size} bytes...")

                    print("✅ Processed gradients sent successfully!")
                    print("🔄 Ready for next batch of gradients...")  # ✅ Keep the connection open for next batch

            except Exception as e:
                print(f"❌ Error: {e}")
                break  

if __name__ == "__main__":
    start_server()
