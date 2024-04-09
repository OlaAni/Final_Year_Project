import HtmlTestRunner
import unittest
import requests

# class TestStringMethods(unittest.TestCase):
#     def test_twoValuesAreEqual(self):
#         value1=True
#         value2=True
#         self.assertEqual(value1, value2)
# if __name__ == '__main__':
#     unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='test_output'))


def saveResult(name, url, result):
    f = open('test.log', 'w+')
    f.write('Test name:' + str(name) + '\n')
    f.write('Test URL:' + str(url) + '\n')
    f.write('Test result:' + str(result) + '\n')
    f.write('---------------------------------------------\n ')
    f.close()


class TestStringMethods(unittest.TestCase):
    def test_twoValuesAreEqual(self):
        value1=True
        value2=True
        self.assertEqual(value1, value2)

def checkServiceForWord(url, keyword):
    try:
        x = requests.get(url)
        print(x.text)
        serverStatus=1
        if keyword in x.text:
            print("found keyword")
            return True
    except:
        print("error")
        return False


url = 'https://jsonplaceholder.typicode.com/todos/1'
result = checkServiceForWord(url, 'userId')
saveResult("find_sim", url, result)



