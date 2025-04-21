apiVersion: v1
kind: Service
metadata:
  name: vdms-service
  labels:
    app: vdms
spec:
  ports:
  - port: 55555
    targetPort: 55555
    protocol: TCP
    name: vdms
  selector:
    app: vdms
---
# TODO: temporary fix!
apiVersion: v1
kind: Service
metadata:
  name: vdms-service-nodeport
  labels:
    app: vdms
spec:
  type: NodePort
  ports:
  - port: 55555
    targetPort: 55555
    nodePort: 30008
    protocol: TCP
    name: vdms
  selector:
    app: vdms
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vdms
  labels:
     app: vdms
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vdms
  template:
    metadata:
      labels:
        app: vdms
    spec:
      enableServiceLinks: false
      containers:
        - name: vdms
          image: intellabs/vdms:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 55555
          volumeMounts:
            - mountPath: /etc/localtime
              name: timezone
              readOnly: true
      volumes:
          - name: timezone
            hostPath:
                path: /etc/localtime
                type: File
