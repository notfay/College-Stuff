package singleton;

public class Database {
	private static Database instance;
	
	private Database() {
		System.out.println("Create Database");
	}
	
	
	public static Database getConnection() {
		
		if (instance == null) {
			synchronized (Database .class) {
				if (instance == null) {
					instance = new Database();
				}
			}
		}
		
		System.out.println("Get Connection");
		return instance;
	}
}
