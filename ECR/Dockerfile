FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ${LAMBDA_TASK_ROOT}

COPY lambda_function.py ${LAMBDA_TASK_ROOT}

COPY pki-root-cert.crt ${LAMBDA_TASK_ROOT}

RUN update-ca-trust extract


RUN pip install -r requirements.txt


CMD [ "lambda_function.lambda_handler" ]
