apiVersion: v1
kind: Service
metadata:
  name: frontend-service
  labels:
    app: frontend
spec:
  type: NodePort
  ports:
    - port: 443
      targetPort: 8443
      nodePort: 30007
      name: https
  selector:
    app: frontend
---
# Part of this could be jobs not deployments.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  labels:
     app: frontend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      enableServiceLinks: false
      containers:
        - name: frontend
          image: defn(`REGISTRY_PREFIX')lcc_frontend:stream
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8443
          envFrom:
            - configMapRef:
                name: proxy-config
          env:
            - name: DBHOST
              value: "vdms-service"
            - name: VDHOST
              value: "http://video-service:8080"
            - name: NO_PROXY
              value: "video-service,$(NO_PROXY)"
            - name: no_proxy
              value: "video-service,$(no_proxy)"
          volumeMounts:
            - mountPath: /etc/localtime
              name: timezone
              readOnly: true
            - mountPath: /var/run/secrets
              name: self-signed-certificate
              readOnly: true
      imagePullSecrets:
        - name:
      volumes:
        - name: timezone
          hostPath:
            path: /etc/localtime
            type: File
        - name: self-signed-certificate
          secret:
            secretName: self-signed-certificate
