import sys
class AsteroidCollision:
    """You have been given n asteroids, which are moving freely in the space. There is a possibility that aster. will collide with each other.
      The aster. are rep. in an array which has pos. and neg. integers. The magnitude of the element showcases the weights of the aster. and the sighs + and - denotes the direction of aster in which they are moving.
        Negative for left and positive for right. When there is a collision btw two aster., if both the aster. have same magnitude, they both collide and they both break. 
        If there is coll. Btw two different magnitude aster. The aster with the samller mag. Will break into pieces. 
    You need to find what all aster. are left in the space after all collision. Velocity is same for all."""


    @staticmethod
    def asteroid_collision_array(asteroids):
        """
        Simulates asteroid collisions using a list (dynamic array) to store the result state.

        This approach uses a list and list methods like append() and pop() 
        to effectively mimic the stack behavior, which is the most natural
        way to handle the collision logic sequentially.

        Args:
            asteroids: A list of integers representing asteroids.
                    Positive -> right, Negative -> left. Magnitude -> size.

        Returns:
            A list (array) of integers representing the asteroids remaining 
            after all collisions.
        """

        result_array = [] 

        for asteroid in asteroids:
            alive = True 
            
            while result_array and asteroid < 0 and result_array[-1] > 0:
                
                last_asteroid_in_result = result_array[-1]
                
                if abs(asteroid) > last_asteroid_in_result:
                    result_array.pop() 
                elif abs(asteroid) == last_asteroid_in_result:
                    result_array.pop()
                    alive = False
                    # The current asteroid 'asteroid' and the last one in result_array explode.
                    break 
                else: # abs(asteroid) < last_asteroid_in_result
                    alive = False
                    # Stop checking collisions for this 'asteroid', move to the next one.
                    break
                    
            
            # If the 'alive' flag is still True, the current asteroid survived 
            if alive:
                result_array.append(asteroid)
                
        return result_array
    

    @staticmethod
    def asteroid_collision_test():
        
        print("Running asteroid collision tests...")
        test_cases = [

            ([5, 10, -5], [5, 10]),
            ([8, -8], []),
            ([10, 2, -5],[10]),
            ([-2, -1, 1, 2], [-2, -1, 1, 2]),
            ([1, -2, -2, -2], [-2, -2, -2]),  
            ([-2, -2, 1, -2], [-2, -2, -2]), 
            ([1, 1, -1, -1], []),          
            ([1, -1, 1, -1], []),         
            ([5, 10, -15], [-15]),        
            ([3, -2, 1, -4, 5], [-4, 5]),     
        ]

        all_passed = True
        for i, (input_case, expected) in enumerate(test_cases):
            result = AsteroidCollision.asteroid_collision_array(list(input_case)) 
            try:
                assert result == expected, f"Test case {i + 1} failed: Input {input_case}, Expected {expected}, Got {result}"
                print(f"Test case {i + 1} PASSED: Input {input_case}, Got {result}")
            except AssertionError as e:
                print(f"Test case {i + 1} FAILED: {e}")
                all_passed = False
            finally:
                sys.stdout.flush()

        print("-" * 20)
        if all_passed:
            print("All asteroid collision tests passed!")
        else:
            print("Some asteroid collision tests failed.")
        print("-" * 20)


if __name__ == "__main__":
    
    AsteroidCollision.asteroid_collision_test() 
