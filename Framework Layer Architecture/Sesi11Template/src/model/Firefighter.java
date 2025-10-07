package model;

public class Firefighter extends Worker{
	
	private int houseOnFire;
	
	public Firefighter(String name, int age) {
		super(name, age);
	}
	
	@Override
	public void work() {
		houseOnFire = random.nextInt(5) + 1;
	}

	@Override
	public void showEarn() {
		// TODO Auto-generated method stub
		int earning = 75000 * houseOnFire;
		System.out.println("Today earnings : " + earning);
		
	}
	
	
}
