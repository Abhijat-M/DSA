class Solution {
    public boolean search(int[] nums, int target) {

        List<Integer> num = new ArrayList<>();

        for (int n : nums) {
            num.add(n);
        }

        return num.contains(target);
        
    }
}