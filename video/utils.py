from queue import Queue

import vdms


class VDMSConnection:
    """Responsible for connecting to VDMS"""

    def __init__(self, dbhost, dbport):
        """Initialize VDMS Connection"""
        self.db = vdms.vdms()
        self.db.connect(dbhost, dbport)

    def get_VDMS_connection(self):
        """Get VDMS client object"""
        return self.db

    def close(self):
        """Close VDMS connection"""
        self.db.disconnect()


class VDMSClientPool:
    """Orchestrates creation, retrieval, and return of connections"""

    def __init__(self, connection_creation, pool_size):
        self.pool = Queue(maxsize=pool_size)
        self.create_connection = connection_creation

        for _ in range(pool_size):
            self.pool.put(self.create_connection())

    def retrieve(self):
        return self.pool.get()

    def release(self, conn):
        """Return to pool"""
        self.pool.put(conn)

    def close_all(self):
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()


class UDFRemotePool:
    """Orchestrates creation, retrieval, and return of udf servers"""

    def __init__(self, pool_size):
        self.pool = Queue(maxsize=pool_size)

        for i in range(pool_size):
            # idx = i + 1
            # self.pool.put(f"udf-service_{idx}")
            self.pool.put("udf-service")

    def retrieve(self):
        return self.pool.get()

    def release(self, udf):
        """Return to pool"""
        self.pool.put(udf)

    def close_all(self):
        while not self.pool.empty():
            _ = self.pool.get()
