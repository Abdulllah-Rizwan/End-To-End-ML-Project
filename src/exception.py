import sys

def get_error_message(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f'Error Occured in Python Script name {file_name} on line number {exc_tb.tb_lineno} error message {str(error)}'
    
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = get_error_message(error_message,error_detail=error_detail)
        
    def __str__(self):
        return self.error_message
    
if __name__ == '__main__':
    a = 10
    b = 0
    try:
        a/b
    except Exception as e:
        raise CustomException(e,sys)   