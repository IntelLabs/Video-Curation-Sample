apiVersion: v1
kind: Service
metadata:
  name: video-service
  labels:
    app: video
spec:
  ports:
    - port: 8080
      targetPort: 8080
      name: http
  selector:
    app: video
---
# TODO: temporary fix!
apiVersion: v1
kind: Service
metadata:
  name: video-service-nodeport
  labels:
    app: video
spec:
  type: NodePort
  ports:
  - port: 8088
    targetPort: 8088
    nodePort: 30009
    protocol: UDP
    name: udp
  selector:
    app: video
---
# Should be a POD not deployment.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: video
  labels:
     app: video
spec:
  replicas: 1
  selector:
    matchLabels:
      app: video
  template:
    metadata:
      labels:
        app: video
    spec:
      enableServiceLinks: false
      containers:
        - name: stream
          image: defn(`REGISTRY_PREFIX')lcc_stream:stream
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8080
            - containerPort: 8088
              protocol: UDP
          envFrom:
            - configMapRef:
                name: proxy-config
          env:
            - name: KKHOST
              value: "kafka-service:9092"
            - name: ZKHOST
              value: "zookeeper-service:2181"
            - name: VDHOST
              value: "http://video-service:8080"
            - name: DBHOST
              value: "vdms-service"
            - name: `STREAM_URL'
              value: "defn(`STREAM_URL')"
            - name: NO_PROXY
              value: "video-service,$(NO_PROXY)"
            - name: no_proxy
              value: "video-service,$(no_proxy)"
          volumeMounts:
            - mountPath: /etc/localtime
              name: timezone
              readOnly: true
            - mountPath: /var/www/mp4
              name: stream-content
              readOnly: false
        - name: video
          image: defn(`REGISTRY_PREFIX')lcc_video:stream
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8080
          envFrom:
            - configMapRef:
                name: proxy-config
          env:
            - name: RETENTION_MINS
              value: "60"
            - name: CLEANUP_INTERVAL
              value: "10m"
            - name: KKHOST
              value: "kafka-service:9092"
            - name: ZKHOST
              value: "zookeeper-service:2181"
            - name: SHOST
              value: "http://stream-service:8080"
            - name: `INGESTION'
              value: "defn(`INGESTION')"
            - name: `IN_SOURCE'
              value: "defn(`IN_SOURCE')"
            - name: NO_PROXY
              value: "stream-service,$(NO_PROXY)"
            - name: no_proxy
              value: "stream-service,$(no_proxy)"
          volumeMounts:
            - mountPath: /etc/localtime
              name: timezone
              readOnly: true
            - mountPath: /var/www/streams
              name: stream-content
              readOnly: false
      imagePullSecrets:
        - name:
      volumes:
        - name: timezone
          hostPath:
            path: /etc/localtime
            type: File
        - name: stream-content
          emptyDir: {}
