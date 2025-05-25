package model;

public class Satpam extends Worker{
	
	public Satpam(String name, int age) {
		super(name, age);
		// TODO Auto-generated constructor stub
	}

	private int orangJahat;

	@Override
	public void work() {
		orangJahat = random.nextInt(5) + 1;
		
	}

	@Override
	public void showEarn() {
		int earnings = 50000 * orangJahat;
		System.out.println("Today earnings : " + earnings);
		
	}
	
	
	
	
}
